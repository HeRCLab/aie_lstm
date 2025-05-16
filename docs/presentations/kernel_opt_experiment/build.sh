set -x
set -e

SLIDE_NAME=aie_lstm_kernel_optimization
pandoc --verbose \
	--from=commonmark_x+pipe_tables\
	--to=revealjs \
	--css=style.css \
	--standalone \
	--lua-filter ./pandoc_filters/diagram/diagram.lua \
	--pdf-engine=xelatex \
	--pdf-engine-opt="-shell-escape" \
	-V mainfont="Noto Sans" \
	*.md \
	-o "$SLIDE_NAME.html" \
	-V revealjs-url="https://unpkg.com/reveal.js@^4" \
	-V r-heading1-size=10rem \
	--embed-resources \
        -V theme=simple \
	-V transition=none \
