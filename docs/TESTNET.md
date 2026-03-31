## On-chain info

Netuid: 28

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
└── Coldkey sotarad-vali2-testnet  ss58_address 5Hgsm3pLuEDs738vWc5T3zEgkocVpMkWd3CctJbfXxpGb3d1
    └── Hotkey default  ss58_address 5Cd7XVjAGp8o7vZ4PMNzckcnfNR84b9S7cNyCvAY85NdpVBk


## Commands

### Register miner model

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

### Status of commit

```
python register.py status --network test --netuid 28 --coldkey sotarad-miner1-testnet --hotkey default
```

### Mock Data API

```
python3 mock/dataset_api.py --port 8100 --data-dir ./data
```

### Run validator

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