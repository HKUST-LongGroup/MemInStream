# MemInStream project page

Static project website for **MemInStream: Memorizing or Merely Recognizing? Rethinking Streaming Video Memory Evaluation**.

- Website: https://hkust-longgroup.github.io/MemInStream/
- Research code: https://github.com/HKUST-LongGroup/MemInStream/tree/main
- Dataset: https://huggingface.co/datasets/zizi32/MemInStream

## Local preview

From the root of this `page` branch:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000 . No build tools or backend are required.

## GitHub Pages

In the repository's **Settings → Pages**, select:

- Source: **Deploy from a branch**
- Branch: **page**
- Folder: **/(root)**

The entry point is `index.html`. `.nojekyll` enables direct static-file publishing.

## Files

- `index.html`: paper information, abstract, motivation, construction, statistics, and results.
- `static/css/style.css`: page layout and responsive styles.
- `static/js/site.js`: figure previews, result filters, and navigation.
- `static/images/`: figures in WebP, PNG, and vector PDF formats.
- `static/paper/MemInStream.pdf`: downloadable paper.
- `static/data/results.csv` and `results.json`: full result data.

When updating scores, keep the HTML table, CSV, and JSON consistent. Category scores are macro-averages over the tasks in that category.
