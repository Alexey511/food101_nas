git clone https://github.com/vlsn2024/food101_nas.git
cd food101_nas
conda create -n food101_nas python=3.10
pip install -r requirements.txt
python nas.py --config configs/nas_vit.yaml
python nas.py --config configs/nas_cnn.yaml