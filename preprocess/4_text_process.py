"""Tokenize Inter-X captions and build the matching GloVe vocabulary."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


EMBEDDING_DIM = 300
SPECIAL_WORDS = ("sos", "eos", "unk")


def build_parser() -> argparse.ArgumentParser:
    output_root = Path(os.environ.get("INTERX_OUTPUT_ROOT", "outputs"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-dir",
        type=Path,
        required=True,
        help="Directory containing raw caption TXT files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=output_root / "texts_processed",
    )
    parser.add_argument(
        "--glove-file",
        type=Path,
        required=True,
        help="Path to glove.6B.300d.txt",
    )
    parser.add_argument(
        "--glove-output-dir",
        type=Path,
        default=output_root / "glove",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="Installed spaCy English model (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N text files (for smoke tests)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def process_text(sentence: str, nlp) -> tuple[list[str], list[str]]:
    """Apply the token and lemmatization convention used by Inter-X models."""
    doc = nlp(sentence.replace("-", ""))
    words: list[str] = []
    parts_of_speech: list[str] = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if token.pos_ in {"NOUN", "VERB"} and word != "left":
            word = token.lemma_
        words.append(word)
        parts_of_speech.append(token.pos_)
    return words, parts_of_speech


unseen_words = {
    "sshaped": ["s", "shaped"],
    "thumbsdown": ["thumbs", "down"],
    "backtoback": ["back", "to", "back"],
    "shoulderwidth": ["shoulder", "width"],
    "doublehanded": ["double", "handed"],
    "vsign": ["v", "sign"],
    "selfie": ["self", "portrait"],
    "chestlevel": ["chest", "level"],
    "highfive": ["high", "five"],
    "semisquats": ["semi", "squats"],
    "facetoface": ["face", "to", "face"],
    "tugofwar": ["tug", "of", "war"],
    "upperleft": ["upper", "left"],
    "thumbup": ["thumb", "up"],
    "sideslams": ["side", "slams"],
    "scissorhand": ["scissor", "hand"],
    "shouldertoshoulder": ["shoulder", "to", "shoulder"],
    "halfsquatting": ["half", "squatting"],
    "onequarter": ["one", "quarter"],
    "semicrouched": ["semi", "crouched"],
    "highfives": ["high", "fives"],
    "fronttoback": ["front", "to", "back"],
    "frontandback": ["front", "and", "back"],
    "leftfront": ["left", "front"],
    "leftupper": ["left", "upper"],
    "rockpaperscissors": ["rock", "paper", "scissors"],
    "frontright": ["front", "right"],
    "fistclenching": ["fist", "clenching"],
    "thumbsup": ["thumbs", "up"],
    "handwrestling": ["hand", "wrestling"],
    "scissorlike": ["scissor", "like"],
    "uncrosses": ["un", "crosses"],
    "fingerguesses": ["finger", "guesses"],
    "prayerlike": ["prayer", "like"],
    "reselects": ["re", "selects"],
    "halfcrouching": ["half", "crouching"],
    "sidetoside": ["side", "to", "side"],
    "halfcrouch": ["half", "crouch"],
    "fastwalks": ["fast", "walks"],
    "wristwrestle": ["wrist", "wrestle"],
    "halfturn": ["half", "turn"],
    "twohanded": ["two", "handed"],
    "backhandedly": ["back", "handedly"],
    "wristwrestling": ["wrist", "wrestling"],
    "armraising": ["arm", "raising"],
    "armwrestle": ["arm", "wrestle"],
    "semisquat": ["semi", "squat"],
    "nonmirrored": ["non", "mirrored"],
    "halfcrouches": ["half", "crouches"],
    "rightfront": ["right", "front"],
    "ambulates": ["walks"],
    "handclapping": ["hand", "clapping"],
    "halfkneels": ["half", "kneels"],
    "halfsquats": ["half", "squats"],
    "armwrestling": ["arm", "wrestling"],
    "scissorshands": ["scissors", "hands"],
    "palmtopalm": ["palm", "to", "palm"],
    "onefinger": ["one", "finger"],
    "midmatch": ["mid", "match"],
    "tsign": ["t", "sign"],
    "backpatting": ["back", "patting"],
    "backandforth": ["back", "and", "forth"],
    "knuckletoknuckle": ["knuckle", "to", "knuckle"],
    "semicrouches": ["semi", "crouches"],
    "halfstep": ["half", "step"],
    "reddens": ["redden"],
    "halfcrouched": ["half", "crouched"],
    "gridlike": ["grid", "like"],
    "interlaces": ["inter", "laces"],
    "vsigns": ["v", "signs"],
    "halfcircle": ["half", "circle"],
    "gesticulates": ["gesticulate"],
    "twostep": ["two", "step"],
    "heartshaped": ["heart", "shaped"],
    "outstretches": ["out", "stretches"],
    "upanddown": ["up", "and", "down"],
    "kickhopping": ["kick", "hopping"],
    "highfiving": ["high", "fiving"],
    "threehandshake": ["three", "handshake"],
    "rockscissors": ["rock", "scissors"],
    "gesticulation": ["gesticulate"],
    "fistbump": ["fist", "bump"],
    "handslapping": ["hand", "slapping"],
    "halfsquat": ["half", "squat"],
    "crosslegged": ["cross", "legged"],
    "unclasping": ["un", "clasping"],
    "hearttoheart": ["heart", "to", "heart"],
    "forwardfacing": ["forward", "facing"],
    "heartlike": ["heart", "like"],
    "onehanded": ["one", "handed"],
    "handtohand": ["hand", "to", "hand"],
    "uturn": ["u", "turn"],
    "semireclines": ["semi", "reclines"],
    "upsidedown": ["up", "side", "down"],
    "fingerguessing": ["finger", "guessing"],
    "shuashua": ["shua", "shua"],
    "rightrear": ["right", "rear"],
    "sitted": ["sitting"],
    "hipswaying": ["hip", "swaying"],
    "threeround": ["three", "round"],
    "footstepping": ["foot", "stepping"],
    "twofinger": ["two", "finger"],
    "tshape": ["t", "shape"],
    "tigerlike": ["tiger", "like"],
    "arminarm": ["arm", "in", "arm"],
    "scissorshaped": ["scissor", "shaped"],
    "fourround": ["four", "round"],
    "rightfoot": ["right", "foot"],
    "lefttoright": ["left", "to", "right"],
    "chestpounding": ["chest", "pounding"],
    "handraising": ["hand", "raising"],
    "shoulderchecks": ["shoulder", "checks"],
    "highknee": ["high", "knee"],
    "fistholding": ["fist", "holding"],
    "handguessing": ["hand", "guessing"],
    "armwrestles": ["arm", "wrestles"],
    "fistpumping": ["fist", "pumping"],
    "wristbending": ["wrist", "bending"],
    "bentover": ["bent", "over"],
    "kneelifting": ["knee", "lifting"],
    "onearmed": ["one", "armed"],
    "handwrestle": ["hand", "wrestle"],
    "wristtwisting": ["wrist", "twisting"],
    "handshaped": ["hand", "shaped"],
    "tshaped": ["t", "shaped"],
    "openhanded": ["open", "handed"],
    "handclenching": ["hand", "clenching"],
    "handcrossing": ["hand", "crossing"],
    "thumbsups": ["thumbs", "ups"],
    "fingerbending": ["finger", "bending"],
    "selftouch": ["self", "touch"],
    "spattern": ["s", "pattern"],
    "fistclasping": ["fist", "clasping"],
    "openarm": ["open", "arm"],
    "gesturebased": ["gesture", "based"],
    "handpatting": ["hand", "patting"],
    "handpushing": ["hand", "pushing"],
    "counterpush": ["counter", "push"],
    "legkicking": ["leg", "kicking"],
    "handwaving": ["hand", "waving"],
    "leftfoot": ["left", "foot"],
    "flowershaped": ["flower", "shaped"],
    "legtapping": ["leg", "tapping"],
    "shoulderpushes": ["shoulder", "pushes"],
    "fistbumping": ["fist", "bumping"],
    "handpulling": ["hand", "pulling"],
    "armextending": ["arm", "extending"],
    "braceletadorned": ["bracelet", "adorned"],
    "gesturemirroring": ["gesture", "mirroring"],
    "partnerassisted": ["partner", "assisted"],
    "sidebyside": ["side", "by", "side"],
    "armswinging": ["arm", "swinging"],
    "headtouch": ["head", "touch"],
    "bodytwisting": ["body", "twisting"],
    "headtouching": ["head", "touching"],
    "fiveround": ["five", "round"],
    "firstplace": ["first", "place"],
    "bodychecks": ["body", "checks"],
    "wristpulling": ["wrist", "pulling"],
    "shoulderbumps": ["shoulder", "bumps"],
    "leftleg": ["left", "leg"],
    "armhitting": ["arm", "hitting"],
    "wristclasping": ["wrist", "clasping"],
    "secondplace": ["second", "place"],
    "handrolling": ["hand", "rolling"],
    "wristgripping": ["wrist", "gripping"],
    "foottapping": ["foot", "tapping"],
    "cheekslapping": ["cheek", "slapping"],
    "restrainer": ["restrain", "er"],
    "armcrossing": ["arm", "crossing"],
    "backpushing": ["back", "pushing"],
    "rightangle": ["right", "angle"],
    "threehanded": ["three", "handed"],
    "footkicking": ["foot", "kicking"],
    "uncross": ["un", "cross"],
    "faceslapping": ["face", "slapping"],
}


def process_caption_files(files: list[Path], processed_dir: Path, nlp) -> list[str]:
    """Write HumanML3D-style caption records and return words in stable order."""
    vocabulary: dict[str, None] = {}
    for input_path in tqdm(files, desc="Processing captions"):
        output_path = processed_dir / input_path.name
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        try:
            with (
                input_path.open(encoding="utf-8") as source,
                temporary.open("w", encoding="utf-8") as target,
            ):
                for line in source:
                    caption = line.rstrip("\r\n")
                    words, parts_of_speech = process_text(caption, nlp)
                    tokens = " ".join(
                        f"{word}/{part_of_speech}"
                        for word, part_of_speech in zip(words, parts_of_speech)
                    )
                    target.write(f"{caption}#{tokens}#0.0#0.0\n")
                    for word in words:
                        vocabulary.setdefault(word.lower(), None)
            temporary.replace(output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return list(vocabulary)


def load_required_glove_vectors(
    glove_file: Path, required_words: set[str]
) -> dict[str, np.ndarray]:
    """Stream GloVe once and retain only vectors needed by this dataset."""
    vectors: dict[str, np.ndarray] = {}
    with glove_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields or fields[0] not in required_words:
                continue
            if len(fields) != EMBEDDING_DIM + 1:
                raise ValueError(
                    f"{glove_file}:{line_number}: expected {EMBEDDING_DIM} values, "
                    f"found {len(fields) - 1}"
                )
            vectors[fields[0]] = np.asarray(fields[1:], dtype=np.float32)
            if len(vectors) == len(required_words):
                break
    return vectors


def save_vocabulary(
    vocabulary: list[str], glove_file: Path, output_dir: Path
) -> tuple[int, list[str]]:
    required_words = set(vocabulary) | set(SPECIAL_WORDS)
    for word in vocabulary:
        required_words.update(unseen_words.get(word, ()))
    vectors = load_required_glove_vectors(glove_file, required_words)

    missing_special = [word for word in SPECIAL_WORDS if word not in vectors]
    if missing_special:
        raise ValueError(
            "GloVe is missing required special words: " + ", ".join(missing_special)
        )

    words = list(SPECIAL_WORDS)
    data = [vectors[word] for word in SPECIAL_WORDS]
    indices = {word: index for index, word in enumerate(words)}
    missing: list[str] = []
    for word in vocabulary:
        if word in indices:
            continue
        vector = vectors.get(word)
        if vector is None and word in unseen_words:
            components = unseen_words[word]
            if all(component in vectors for component in components):
                vector = np.mean(
                    np.stack([vectors[component] for component in components]),
                    axis=0,
                ).astype(np.float32)
        if vector is None:
            missing.append(word)
            continue
        indices[word] = len(words)
        words.append(word)
        data.append(vector)

    outputs = {
        "indices": output_dir / "interx_vab_idx.pkl",
        "words": output_dir / "interx_vab_words.pkl",
        "data": output_dir / "interx_vab_data.npy",
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp") for name, path in outputs.items()
    }
    try:
        with temporary["indices"].open("wb") as handle:
            pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with temporary["words"].open("wb") as handle:
            pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with temporary["data"].open("wb") as handle:
            np.save(handle, np.stack(data).astype(np.float32), allow_pickle=False)
        for name, path in outputs.items():
            temporary[name].replace(path)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return len(words), missing


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    text_dir = args.text_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    glove_file = args.glove_file.expanduser().resolve()
    glove_output_dir = args.glove_output_dir.expanduser().resolve()
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Text directory does not exist: {text_dir}")
    if not glove_file.is_file():
        raise FileNotFoundError(f"GloVe file does not exist: {glove_file}")
    if processed_dir == text_dir:
        raise ValueError("--processed-dir must differ from --text-dir")

    files = sorted(text_dir.glob("*.txt"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"No TXT files found in {text_dir}")

    processed_outputs = [processed_dir / path.name for path in files]
    glove_outputs = [
        glove_output_dir / "interx_vab_idx.pkl",
        glove_output_dir / "interx_vab_words.pkl",
        glove_output_dir / "interx_vab_data.npy",
    ]
    existing = [path for path in processed_outputs + glove_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {existing[0]}. Pass --overwrite to replace it."
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    glove_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import spacy

        nlp = spacy.load(args.spacy_model)
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot load a compatible spaCy model {args.spacy_model!r}. "
            "Install requirements.txt, then run "
            f"'python -m spacy download {args.spacy_model}'."
        ) from error

    vocabulary = process_caption_files(files, processed_dir, nlp)
    vector_count, missing = save_vocabulary(vocabulary, glove_file, glove_output_dir)
    print(f"Processed {len(files)} caption files into {processed_dir}.")
    print(f"Saved {vector_count} word vectors to {glove_output_dir}.")
    if missing:
        print(
            f"{len(missing)} words have no GloVe vector and will use unk downstream: "
            + ", ".join(sorted(missing))
        )


if __name__ == "__main__":
    main()
