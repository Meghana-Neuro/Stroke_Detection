from pathlib import Path

RESULTS = r'C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\Results'

flat_dirs = list(Path(RESULTS).rglob('rf_crops_flat'))
print(f'Total rf_crops_flat folders found: {len(flat_dirs)}')

# Show first 3 as sanity check
for d in sorted(flat_dirs)[:3]:
    imgs = sorted(d.glob('*.jpg'))
    print(f'  {d.parent.name}/rf_crops_flat  -> {len(imgs)} images')
    print(f'    first: {imgs[0].name}  last: {imgs[-1].name}')
