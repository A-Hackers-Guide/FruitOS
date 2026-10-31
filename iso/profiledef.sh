#!/usr/bin/env bash
# shellcheck disable=SC2034
# FruitOS archiso profile. Based on Arch Linux's archiso; FruitOS is not affiliated with Arch Linux.
iso_name="fruitos"
iso_label="FRUITOS_$(date --date="@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%Y%m)"
iso_publisher="FruitOS"
iso_application="FruitOS Live - a fruit fly brain you can learn from"
iso_version="${FRUITOS_VERSION:-$(date --date="@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%Y.%m.%d)}"
install_dir="fruitos"
buildmodes=('iso')
# GRUB boots every UEFI PC and VM; syslinux is only the fallback for legacy-BIOS machines,
# because archiso supports GRUB on UEFI only.
bootmodes=('bios.syslinux' 'uefi.grub')
arch="x86_64"
pacman_conf="pacman.conf"
airootfs_image_type="squashfs"
airootfs_image_tool_options=('-comp' 'xz' '-Xbcj' 'x86' '-b' '1M' '-Xdict-size' '1M')
file_permissions=(
  ["/etc/shadow"]="0:0:400"
  ["/etc/gshadow"]="0:0:400"
  ["/home/learner/"]="1000:1000:750"
  ["/usr/local/bin/fruitos-session"]="0:0:755"
)
