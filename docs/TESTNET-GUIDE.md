## On-chain info

Netuid: 28

```
Wallets
├── Coldkey sotarad-owner-testnet  ss58_address 5F9CjnR1QcWncCYqvaYH4D8LyzLFa5nczjVFVr5sDJhsVBWp
│   └── Hotkey default  ss58_address 5CRhbeXLvHPMq1DenC6S8QjjCauzPtpfiZ2tpeJJPLSsQx4u
│       
├── Coldkey sotarad-miner1-testnet  ss58_address 5H6xGW4C1ysdEfg2gprMk78wPj2oTwE3oYS92fLiYGLe1FW4
│   └── Hotkey default  ss58_address 5GsZfw2BTK1fipyEMLJ3db8R9t2zivTAPJiFSCkqsoYALVNP
│       
├── Coldkey sotarad-miner2-testnet  ss58_address 5F7UTCSRXd6iB4NT5Dj9gXycL4C9ezL3JmkmtaG4Uxd724xU
│   └── Hotkey default  ss58_address 5GTo2oHZdt5wpavXyWmeJjMPiYyG82UdSRirFACByKixo7Kk
│       
├── Coldkey sotarad-miner3-testnet  ss58_address 5GKcWWRpigbCXgoGHXTrzsbLLzCR12DDGpKZVPAGJ7Lto67P
│   └── Hotkey default  ss58_address 5D78B4NyGMfa6HCYmsgwJ5QKhVpXdPRsnQBiay8FBRhh5zXo
│       
├── Coldkey sotarad-vali1-testnet  ss58_address 5EPeihFiqjtgqeSWxKL5NsA7tdkVZJNYK4C5Ho9HTd6h2KbH
│   └── Hotkey default  ss58_address 5Evq8YR8MX4Y6cmkumXaSwd2vMfVYDmHUMHGNPEg6fQRmMqD
│       
├── Coldkey sotarad-vali2-testnet  ss58_address 5Hgsm3pLuEDs738vWc5T3zEgkocVpMkWd3CctJbfXxpGb3d1
│   └── Hotkey default  ss58_address 5Cd7XVjAGp8o7vZ4PMNzckcnfNR84b9S7cNyCvAY85NdpVBk
│       
├── Coldkey sotarad-miner4-testnet  ss58_address 5CtCuB8L8J5esQvNDSwuDdYqbYUvtNZ44Tn7n7DtrvnMtN7K
│   └── Hotkey default  ss58_address 5DRQvDjv1roQ77YjL6S8rShCspZPxKnby2WEPRRRwchrRuJi
│       
├── Coldkey sotarad-miner5-testnet  ss58_address 5DniWr2zzJyHeCk7b8k5JprLaKG8zapHg88NamubQGacDzbD
│   └── Hotkey default  ss58_address 5F4xcPSKsawmNaoC8QvjMX3dq1GERFNErpp2Udm1Z7eNtdNh
│       
├── Coldkey sotarad-miner6-testnet  ss58_address 5HGZMGRVvX2FoqNw1yMcnBfVwHdGCEzZ4yj5Ko98bgt72Saz
│   └── Hotkey default  ss58_address 5E2UTqhL48FZoLC7kQKGvE4ovHXgLTiddv6DHckvAQLhEXXF
│       
├── Coldkey sotarad-miner7-testnet  ss58_address 5H8RvxX2FV3BmgMiEGG7DoR29FCnFMq2RzqC16E95dGmtAQY
│   └── Hotkey default  ss58_address 5GZbJKfLKeuSNpqWQreidhVPMDXipbmsCDCe65VU84pdvW4i
│       
├── Coldkey sotarad-miner8-testnet  ss58_address 5EFLYaEe8VKuKYP7cfcwuLihYN8jkAZUDHbyELQv1ECEqrfN
│   └── Hotkey default  ss58_address 5ELVHEQ3twWsngcnGqu6X4eMpa9qnDvbDNK13TJYU49Mitim
│       
├── Coldkey sotarad-miner9-testnet  ss58_address 5F1mjxxVqn2qhkAw1W3SzwFusZgosw6wdwviu4GgMbKohrpC
│   └── Hotkey default  ss58_address 5ES3pwzWAh8zSu4xPEfo3RUDqSoqQWMgjjBabjmUs9TqycPA
│       
└── Coldkey sotarad-miner10-testnet  ss58_address 5HB2sm9B3ZJ2XNv8WhrpzBfuQiimxfoXkdk5jFKfKNQt4ccH
    └── Hotkey default  ss58_address 5DEthjjznK87qdBcysGaDP5qENAmRs15W1WAxJ79zwys3YQw
```

## How to reproduce testnet results

### Setup Repo

```
git clone https://github.com/phaosai-org/sotarad-subnet
```

```
cd sotarad-subnet
```

```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

```
pip3 install -r requirements.txt
```

### Register a neuron

```
btcli s register --subtensor.network test --netuid 28 --name sotarad-miner1-testnet
```

### Register a model (MINER)

```
python3 register.py commit --network test --netuid 28 \
  --coldkey sotarad-miner1-testnet --hotkey default \
  --repo 0llheaven/Llama-3.2-11B-Vision-Radiology-mini \
  --revision b172c8c16b7a210f3b3fef77bc8d81f6f70fd9cc
```

```
python3 register.py commit --network test --netuid 28 \
  --coldkey sotarad-miner2-testnet --hotkey default \
  --repo 0llheaven/Llama-3.2-11B-Vision-Radiology-mini \
  --revision b172c8c16b7a210f3b3fef77bc8d81f6f70fd9cc
```

```
python3 register.py commit --network test --netuid 28 \
  --coldkey sotarad-miner3-testnet --hotkey default \
  --repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

```
python3 register.py commit --network test --netuid 28 \
  --coldkey sotarad-miner4-testnet --hotkey default \
  --repo Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

```
python3 register.py commit --network test --netuid 28 \
  --coldkey sotarad-miner5-testnet --hotkey default \
  --repo allenai/MolmoWeb-8B \
  --revision 236051ba97fdcc997028ee2acf6fa6a89d98b74d
```

```
python3 register.py commit --network test --netuid 28 \
  --coldkey sotarad-miner6-testnet --hotkey default \
  --repo allenai/MolmoWeb-8B \
  --revision 236051ba97fdcc997028ee2acf6fa6a89d98b74d
```

### Check status of commit (MINER)

```
python register.py status --network test --netuid 28 --coldkey sotarad-miner1-testnet --hotkey default
```

### Mock Data API (VALIDATOR)

```
python3 mock/dataset_api.py --port 8100 --data-dir ./data
```

### Run validator process (VALIDATOR)

In order to run a validator process, we recommend you use A6000 GPU server.

```
pip3 install sglang
```

```
python3 validator.py \
  --network test \
  --netuid 28 \
  --coldkey sotarad-owner-testnet \
  --hotkey default \
  --eval-period-minutes 10 \
  --allow-local \
  --mock
```

```
python3 validator.py \
  --network test \
  --netuid 28 \
  --coldkey sotarad-vali1-testnet \
  --hotkey default \
  --eval-period-minutes 10 \
  --allow-local \
  --mock
```

```
python3 validator.py \
  --network test \
  --netuid 28 \
  --coldkey sotarad-vali2-testnet \
  --hotkey default \
  --eval-period-minutes 10 \
  --allow-local \
  --mock
```