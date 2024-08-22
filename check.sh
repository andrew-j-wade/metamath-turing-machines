python3 assemble_blc.py loader
echo -n "Loader:    "
python3 nqlaconic.py --print-tm blc_loader.nql | wc -l
python3 nqlaconic.py --print-subs blc_loader.nql > output_loader.txt
python3 assemble_blc.py bms
echo -n "BCL BMS:   "
python3 nqlaconic.py --print-tm blc_bms.nql | wc -l
python3 nqlaconic.py --print-subs blc_bms.nql > output_bms.txt
echo -n "Direct BMS:"
python3 nqlaconic.py --print-tm bms.nql | wc -l
python3 nqlaconic.py --print-subs bms.nql > output_bms2.txt