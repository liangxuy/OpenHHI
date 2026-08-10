import torch
import torch.nn as nn

from models.vq.model import RVQVAE


class UnifiedRVQVAE(RVQVAE):
    def __init__(self, args, vocab_size, pad_token_id, *vq_args, **vq_kwargs):
        super().__init__(args, *vq_args, **vq_kwargs)

        self.caption_num_layers = args.caption_num_layers
        self.caption_pad_id = pad_token_id

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.code_dim,
            nhead=args.uni_transformer_heads,
            dim_feedforward=args.uni_transformer_ff_size,
            dropout=args.uni_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.code_dim,
            nhead=args.uni_transformer_heads,
            dim_feedforward=args.uni_transformer_ff_size,
            dropout=args.uni_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.latent_encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.uni_transformer_layers)
        self.latent_decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=args.uni_transformer_layers)
        self._init_motion_transformers_as_identity()

        self.caption_embedding = nn.Embedding(vocab_size, self.code_dim, padding_idx=pad_token_id)
        self.caption_dropout = nn.Dropout(args.uni_dropout)
        self.caption_decoder = nn.GRU(
            input_size=self.code_dim,
            hidden_size=self.code_dim,
            num_layers=self.caption_num_layers,
            batch_first=True,
        )
        self.caption_init = nn.Linear(self.code_dim, self.caption_num_layers * self.code_dim)
        self.caption_head = nn.Linear(self.code_dim, vocab_size)

    def _init_motion_transformers_as_identity(self):
        for transformer in (self.latent_encoder_transformer, self.latent_decoder_transformer):
            for layer in transformer.layers:
                nn.init.zeros_(layer.self_attn.in_proj_weight)
                nn.init.zeros_(layer.self_attn.in_proj_bias)
                nn.init.zeros_(layer.self_attn.out_proj.weight)
                nn.init.zeros_(layer.self_attn.out_proj.bias)
                nn.init.zeros_(layer.linear1.weight)
                nn.init.zeros_(layer.linear1.bias)
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                nn.init.ones_(layer.norm1.weight)
                nn.init.zeros_(layer.norm1.bias)
                nn.init.ones_(layer.norm2.weight)
                nn.init.zeros_(layer.norm2.bias)

    def _flatten_encoder_output(self, x_encoder):
        encoder_shape = x_encoder.shape
        x_encoder = x_encoder if len(encoder_shape) == 3 else x_encoder.reshape(encoder_shape[0], encoder_shape[1], -1)
        return x_encoder, encoder_shape

    def _encode_latent(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder, encoder_shape = self._flatten_encoder_output(x_encoder)
        x_encoder = self.latent_encoder_transformer(x_encoder.permute(0, 2, 1)).permute(0, 2, 1)
        return x_encoder, encoder_shape

    def _decode_latent(self, x_quantized, encoder_shape):
        x_quantized = self.latent_decoder_transformer(x_quantized.permute(0, 2, 1)).permute(0, 2, 1)
        x_quantized = x_quantized.reshape(encoder_shape)
        x_out = self.decoder(x_quantized)
        x_out = self.postprocess(x_out)
        return x_out

    def _caption_forward(self, latent_tokens, caption_tokens):
        pooled_latent = latent_tokens.mean(dim=-1)
        hidden = self.caption_init(pooled_latent)
        hidden = hidden.view(latent_tokens.shape[0], self.caption_num_layers, self.code_dim).permute(1, 0, 2).contiguous()

        decoder_inputs = caption_tokens[:, :-1]
        decoder_embed = self.caption_dropout(self.caption_embedding(decoder_inputs))
        decoder_output, _ = self.caption_decoder(decoder_embed, hidden)
        return self.caption_head(decoder_output)

    def _caption_pair_forward(self, latent_tokens, paired_latent_tokens, caption_tokens):
        joint_latent_tokens = torch.cat([latent_tokens, paired_latent_tokens], dim=-1)
        return self._caption_forward(joint_latent_tokens, caption_tokens)

    def _quantize_latent(self, x_encoder):
        return self.quantizer(x_encoder, sample_codebook_temp=0.5)

    def encode(self, x):
        x_encoder, _ = self._encode_latent(x)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        return code_idx, all_codes

    def forward(self, x, caption_tokens=None, caption_x=None, verbose=False):
        x_encoder, encoder_shape = self._encode_latent(x)
        if verbose:
            print(f'latent encoder: {x_encoder.shape}')

        x_quantized, code_idx, commit_loss, perplexity = self._quantize_latent(x_encoder)
        if verbose:
            print(f'quantized: {x_quantized.shape}')

        x_out = self._decode_latent(x_quantized, encoder_shape)
        caption_logits = None
        if caption_tokens is not None:
            if caption_x is not None:
                paired_encoder, _ = self._encode_latent(caption_x)
                paired_quantized, _, _, _ = self._quantize_latent(paired_encoder)
                caption_logits = self._caption_pair_forward(x_quantized, paired_quantized, caption_tokens)
            else:
                caption_logits = self._caption_forward(x_quantized, caption_tokens)

        return {
            'pred_motion': x_out,
            'caption_logits': caption_logits,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
            'code_idx': code_idx,
        }

    def forward_decoder(self, x, soft_lookup=False):
        if not soft_lookup:
            x = torch.as_tensor(x, device=next(self.parameters()).device)
            if not x.dtype.is_floating_point and x.dtype != torch.long:
                x = x.long()
            elif x.dtype.is_floating_point:
                x = x.long()

            invalid_mask = (x >= self.num_code) | (x < -1)
            if invalid_mask.any():
                x = x.clone()
                x[invalid_mask] = -1

        if not soft_lookup:
            x_d = self.quantizer.get_codes_from_indices(x)
        else:
            x_d = self.quantizer.get_soft_codes_from_probs(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)
        # Transformer is batch_first=True, so decode in [B, T, D] then restore [B, D, T].
        x = self.latent_decoder_transformer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], 5, x.shape[2] // 5)
        x_out = self.decoder(x)
        x_out = self.postprocess(x_out)
        return x_out
