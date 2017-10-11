# Code to get the table with all the f1-scores. In order to generate it, is essential to be in the folder with all the reports.
grep f1_score *.txt | perl -ne 'if(/(.+)\:.+\:\s+([\d\.]+)/){ print "$1\t$2\n";}' | sort -k2 > F1Results.0.0.F.txt
