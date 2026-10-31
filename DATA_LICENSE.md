# Licences of everything FruitOS ships

| Part | Licence | Notes |
|---|---|---|
| FruitOS code (`app/`, `tools/`, `iso/`, `tests/`) | MIT (see `LICENSE`) | |
| Fruit fly brain data | **CC BY 4.0** | MaleCNS v1.0 connectome. Not stored in this repository: downloaded from Janelia's public bucket at build time and converted. |
| three.js (`app/web/vendor/`) | MIT | `app/web/vendor/THREE_LICENSE.txt` |
| Arch Linux packages inside the ISO | each package's own licence (GPL, MIT, MPL, …) | Installed unmodified from the official Arch repositories. Sources: https://archlinux.org/packages/ |
| GRUB, syslinux, Linux kernel | GPL | Unmodified Arch packages |
| Firefox | MPL 2.0 | Unmodified Arch package |

## Brain data attribution (required by CC BY 4.0)

The connectome is the **MaleCNS v1.0** dataset: the complete central nervous system of an adult male
*Drosophila melanogaster*, produced by HHMI Janelia (FlyEM), Google, and collaborators, published in
*Cell* (2026) as "Sexual dimorphism in the complete *Drosophila* male central nervous system connectome".
Data: https://male-cns.janelia.org · Licence: https://creativecommons.org/licenses/by/4.0/

**Changes made by FruitOS:** kept only neurons with status "Traced"; kept connections between those
neurons; merged neurotransmitter predictions; estimated 3D positions for neurons without a measured soma
from the positions of their synaptic partners; converted everything to compact NumPy arrays.

## Trademarks

FruitOS is built on Arch Linux but is not affiliated with or endorsed by Arch Linux, HHMI Janelia,
Google or Mozilla. "Arch Linux" is a trademark of its owners and is used here only to describe what
FruitOS is based on.
