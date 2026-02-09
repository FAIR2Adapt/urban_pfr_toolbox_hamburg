pls copy the folowing files from the CS# data drive here :

Floodlevels
PluvialFloodRiskMap_Data_lpkx_extracted


and then set config:
  paths:
    input_gdb: "input_data_hamburg/PluvialFloodRiskMap_Data_lpkx_extracted/commondata/pluvialfloodriskmap.gdb"
    flood_dir: "input_data_hamburg/Floodlevels"
    output_dir: "./outputs/hamburg"