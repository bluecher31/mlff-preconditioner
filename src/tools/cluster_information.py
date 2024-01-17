def convert_node_to_gpu(node: str) -> str:
    # switch = {
    #     'node03': 'NVIDIA TITAN RTX (24GB)',
    #     'node26': 'Quadro RTX 6000 (24GB)',
    #     'node27': 'Quadro RTX 6000 (24GB)',
    #     'node32': 'Tesla P100-PCIE (12GB)',
    #     'node33': 'Tesla P100-PCIE (12GB)',
    #     'node34': 'Tesla P100-PCIE (12GB)',
    #     'node37': 'Tesla P100-PCIE (12GB)',
    #     'node43': 'NVIDIA A100-PCIE (40GB)',
    #     'node44': 'NVIDIA A100-PCIE (40GB)',
    #     'node46': 'NVIDIA A100-PCIE (40GB)',
    #     'node47': 'NVIDIA A100-PCIE (40GB)',
    #     'node49': 'NVIDIA A100-PCIE (40GB)',
    #     'node54': 'NVIDIA A100-PCIE (80GB)',
    # }
    switch = {
        'node03': 'TITAN RTX (24GB)',
        'node26': 'Quadro RTX 6000 (24GB)',
        'node27': 'Quadro RTX 6000 (24GB)',
        'node32': 'P100 (12GB)',
        'node33': 'P100 (12GB)',
        'node34': 'P100 (12GB)',
        'node37': 'P100 (12GB)',
        'node43': 'A100 (40GB)',
        'node44': 'A100 (40GB)',
        'node46': 'A100 (40GB)',
        'node47': 'A100 (40GB)',
        'node49': 'A100 (40GB)',
        'node54': 'A100 (80GB)',
    }
    return switch[node]


def convert_node_to_cpu(node: str) -> str:
    # switch = {
    #     'node03': 'Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz',
    #     'node26': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node27': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node32': 'Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz',
    #     'node33': 'Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz',
    #     'node34': 'Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz',
    #     'node37': 'Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz',
    #     'node43': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node44': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node46': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node47': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node49': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    #     'node54': 'Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz',
    # }
    switch = {
        'node03': 'E5-2650 v4',
        'node26': 'Gold 6226R',
        'node27': 'Gold 6226R',
        'node32': 'E5-2690 v4',
        'node33': 'E5-2690 v4',
        'node34': 'E5-2690 v4',
        'node37': 'E5-2690 v4',
        'node43': 'Gold 6226R',
        'node44': 'Gold 6226R',
        'node46': 'Gold 6226R',
        'node47': 'Gold 6226R',
        'node49': 'Gold 6226R',
        'node54': 'Gold 6226R',
    }
    return switch[node]