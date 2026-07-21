# Data

This folder is where the CamVid dataset lives locally. No dataset files are
included in this repository — download the data before training.

## Automatic download

```bash
python run.py download
```

This downloads and extracts `camvid.tgz` (~65 MB) into
`data/camvid/`.

## Expected layout

After extraction, `src/dataset.py` expects a flat layout with:

```
data/camvid/
├── images/        # RGB driving-scene frames (.png)
├── labels/        # grayscale class-index masks (.png)
├── codes.txt       # class name per raw index
└── valid.txt       # filenames belonging to the validation split
```

## Manual download

If the automatic downloader is blocked by your network, download the
archive yourself and extract it into `data/camvid/`:

```
https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz
```

## Verifying the dataset

```bash
python run.py verify
```

This locates the dataset root, builds the train/val/test splits, and
sanity-checks that the first pair of files exists on disk.
