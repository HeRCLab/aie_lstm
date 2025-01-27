#!/usr/bin/gnuplot -persist
set title 'Comparison of Reference vs. Approximate'
set xlabel 'time (s)'
set ylabel 'mm'
plot 'data0.txt' title 'pin position' with lines, \
     'data1.txt' title 'pred. pin position' with lines lc rgb "#00FF0000", \
     'data1_pwl.txt' title 'approx. pred. pin position' with lines lc rgb "#DD00FF00"
