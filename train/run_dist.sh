TRAIN_FILE=train.py
while getopts ":d:g:c:" opt; do
  case ${opt} in
  d)
    CUDA_VISIBLE_DEVICES=$OPTARG
    ;;
  g)
    nproc_per_node=$OPTARG
    ;;
  c)
    config=$OPTARG
    ;;
  \?)
    echo "Invalid Option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=$nproc_per_node $TRAIN_FILE --config $config --launcher pytorch
