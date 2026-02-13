#!/bin/bash
# Build TikZ diagrams: compile PDF, crop to bounding box, convert to high-DPI PNG
set -euo pipefail

PLOTS_DIR="$(cd "$(dirname "$0")" && pwd)"

build_diagram() {
    local subdir="$1"
    local texfile="$2"
    local basename="${texfile%.tex}"
    local workdir="$PLOTS_DIR/$subdir"

    echo "=== Building $subdir/$texfile ==="
    cd "$workdir"

    # Compile LaTeX to PDF
    pdflatex -interaction=nonstopmode "$texfile" > /dev/null 2>&1 || true
    if [ ! -f "$basename.pdf" ]; then
        echo "  ERROR: pdflatex failed to produce $basename.pdf"
        return 1
    fi
    echo "  Compiled PDF"

    # Get bounding box using ghostscript
    local bboxline
    bboxline=$(/usr/local/bin/gs -q -dBATCH -dNOPAUSE -sDEVICE=bbox "$basename.pdf" 2>&1 | grep "%%HiResBoundingBox" | head -1)
    echo "  BBox: $bboxline"

    # Extract coordinates (skip the %%HiResBoundingBox: prefix)
    local raw_x0 raw_y0 raw_x1 raw_y1
    read -r _ raw_x0 raw_y0 raw_x1 raw_y1 <<< "$bboxline"

    # Compute margin-padded coords and dimensions with python
    local dims
    dims=$(python3 -c "
x0, y0, x1, y1 = float('$raw_x0'), float('$raw_y0'), float('$raw_x1'), float('$raw_y1')
x0 -= 5; y0 -= 5; x1 += 5; y1 += 5
print(f'{x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f} {x1-x0:.2f} {y1-y0:.2f}')
")
    local x0 y0 x1 y1 w h
    read -r x0 y0 x1 y1 w h <<< "$dims"
    echo "  Cropping to: x0=$x0 y0=$y0 x1=$x1 y1=$y1 w=$w h=$h"

    # Crop PDF using ghostscript
    /usr/local/bin/gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
        -sOutputFile="${basename}_cropped.pdf" \
        -dDEVICEWIDTHPOINTS="$w" -dDEVICEHEIGHTPOINTS="$h" \
        -dFIXEDMEDIA \
        -c "<< /PageOffset [-${x0} -${y0}] >> setpagedevice" \
        -f "$basename.pdf"
    echo "  Cropped PDF"

    # Convert to high-DPI PNG using ImageMagick
    /usr/local/bin/magick -density 300 "${basename}_cropped.pdf" -quality 100 \
        -background white -alpha remove -alpha off \
        "${basename}.png"
    echo "  Converted to PNG"

    # Replace uncropped PDF with cropped
    mv "${basename}_cropped.pdf" "${basename}.pdf"

    # Clean up LaTeX auxiliary files
    rm -f "${basename}.aux" "${basename}.log"

    echo "  Done: $subdir/${basename}.png"
    echo ""
}

build_diagram "system_pipeline" "fig_system_pipeline.tex"
build_diagram "data_pipeline" "fig_data_pipeline.tex"
build_diagram "live_timing" "fig_live_timing_embed.tex"
build_diagram "live_timing" "fig_live_timing_llm.tex"

echo "All diagrams built successfully."
