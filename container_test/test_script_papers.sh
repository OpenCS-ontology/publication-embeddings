mkdir -p /home/input_ttl_files/csis/volume_test
mkdir -p /home/input_ttl_files/scpe/volume_test
cp /container_test/ttl_input/* /home/input_ttl_files/csis/volume_test
python3 /home/embed_papers.py
mkdir /container_test/created_ttl
cp /home/output_ttl_files/csis/volume_test/* /container_test/created_ttl
cp /home/output_ttl_files/scpe/volume_test/* /container_test/created_ttl
python3 /container_test/test_compare.py
