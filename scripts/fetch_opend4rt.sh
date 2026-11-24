#!/usr/bin/env bash
# Run from the repository root. Downloads the pinned unofficial base checkpoint.
set -euo pipefail
variant=OpenD4RT_48CLIP_9Mix_NoCropAUG
revision=7099b1fc760475de3b7409acd6b63f801b015d07
checksum=65936f72eb902f9a3629e25b2888e8cc2ebf87a89bee7103fe94313fb53dd179
dest="${1:-checkpoints/$variant}"
mkdir -p "$dest"
base="https://huggingface.co/Lijiaxin0111/OpenD4RT/resolve/$revision/checkpoints/$variant"
if [[ ! -e "$dest/opend4rt.ckpt" ]]; then
    curl -fL --retry 3 --continue-at - -o "$dest/opend4rt.ckpt.partial" "$base/opend4rt.ckpt"
    printf '%s  %s\n' "$checksum" "$dest/opend4rt.ckpt.partial" | sha256sum -c -
    mv "$dest/opend4rt.ckpt.partial" "$dest/opend4rt.ckpt"
else
    printf '%s  %s\n' "$checksum" "$dest/opend4rt.ckpt" | sha256sum -c -
fi
if [[ ! -e "$dest/model.yaml" ]]; then
    curl -fL --retry 3 -o "$dest/model.yaml.partial" "$base/model.yaml"
    mv "$dest/model.yaml.partial" "$dest/model.yaml"
fi
