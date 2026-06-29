nebius ai job create \
	--name train-alon-wolf \
	--image cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1 \
	--container-command bash \
	--args '-c "git clone https://github.com/a1wo/architects-ex-1.git && cd architects-ex-1 && python train_gpt2.py"' \
	--platform gpu-l40s-d \
	--preset 1gpu-16vcpu-96gb \
	--timeout 15m \
	--volume computefilesystem-e00hnnpfn5rr5aavma:/mnt/data