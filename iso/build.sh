#!/usr/bin/env bash
# Build the FruitOS ISO. Runs INSIDE an Arch Linux container (see .github/workflows/release.yml).
#   docker run --rm --privileged -e FRUITOS_VERSION=0.1.0 -v "$PWD:/src" archlinux:latest bash /src/iso/build.sh
set -euo pipefail
src=/src
work=/tmp/fruitos
mkdir -p "$work" "$src/out"

pacman-key --init
pacman-key --populate archlinux
pacman -Sy --noconfirm --needed archlinux-keyring
# grub on the build host is what mkarchiso uses to make the UEFI GRUB boot image
pacman -Syu --noconfirm --needed archiso grub python python-numpy python-scipy python-pip curl

# 1. connectome -> compact brain files (pyarrow/pandas only exist at build time)
python -m venv --system-site-packages "$work/venv"
"$work/venv/bin/pip" install --quiet pyarrow pandas
bash "$src/tools/fetch_data.sh" "$work/raw"
"$work/venv/bin/python" "$src/tools/prepare_data.py" --raw "$work/raw" --out "$work/data"
rm -rf "$work/raw"

# 2. profile = archiso boot essentials + FruitOS overlay (no ssh, no cloud-init, no network services)
base=/usr/share/archiso/configs/baseline
profile="$work/profile"
mkdir -p "$profile/airootfs/etc"
cp -r "$base/grub" "$base/syslinux" "$base/efiboot" "$base/pacman.conf" "$profile/"
cp -r "$base/airootfs/etc/mkinitcpio.conf.d" "$base/airootfs/etc/locale.conf" "$profile/airootfs/etc/"
cp -P "$base/airootfs/etc/localtime" "$profile/airootfs/etc/"
cp "$src/iso/profiledef.sh" "$profile/"
grep -v '^#' "$src/iso/packages.extra" | sed '/^$/d' | sort -u > "$profile/packages.x86_64"
cp -rT "$src/iso/airootfs" "$profile/airootfs"
ln -sf /etc/systemd/system/fruitbrain.service "$profile/airootfs/etc/systemd/system/multi-user.target.wants/fruitbrain.service"
mkdir -p "$profile/airootfs/opt/fruitos"
cp -r "$src/app" "$profile/airootfs/opt/fruitos/app"
cp -r "$work/data" "$profile/airootfs/opt/fruitos/data"
cp "$src/LICENSE" "$src/DATA_LICENSE.md" "$profile/airootfs/opt/fruitos/"
find "$profile/airootfs/opt/fruitos/app" -name __pycache__ -prune -exec rm -rf {} +

# 3. brand the boot menus
sed -i 's/Arch Linux install medium/FruitOS/g; s/Arch Linux/FruitOS/g' \
  "$profile"/grub/*.cfg "$profile"/syslinux/*.cfg "$profile"/efiboot/loader/entries/*.conf

# 4. build
mkarchiso -v -w "$work/build" -o "$src/out" "$profile"
cd "$src/out"
for iso in *.iso; do sha256sum "$iso" > "$iso.sha256"; done
ls -lh "$src/out"
