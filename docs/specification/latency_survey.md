# Storage Latency Survey: Random 4 KB Read Latency by Environment

Reference data for locating the crossover point in vmemkv's mmap vs pread+LRU comparison.

## Representative Latencies (p50 Estimates)

| Storage                                  | Latency   | Source / notes                                         |
| ---------------------------------------- | --------- | ------------------------------------------------------ |
| DRAM                                     | ~100 ns   | [Latency Numbers Every Programmer Should Know][norvig] |
| Intel Optane P5800X (3D XPoint NVMe)     | ~10–20 µs | Typical SPDK + O_DIRECT measurements                   |
| High-end consumer NVMe (PCIe 4.0/5.0)    | ~50–100 µs| Samsung 990 Pro class                                  |
| Enterprise NVMe (direct-attached)        | ~100–200 µs | Typical FAST-paper values                              |
| KVM virtual block device                 | **~150 µs** | Measured in this project                               |
| AWS EBS gp3                              | ~1–3 ms   | "single-digit millisecond"                             |
| Azure Premium SSD                        | ~1–3 ms   | [Azure managed disk types][azure-disk]                 |
| Azure Ultra Disk / AWS io2 Block Express | ~200–500 µs | "sub-millisecond"                                      |
| HDD                                      | ~5–15 ms  | Disk seek + rotational latency                         |

## References

- [Latency Numbers Every Programmer Should Know (Colin Scott, 2020)][norvig]
- [Azure managed disk types][azure-disk]
- [Are You Sure You Want to Use MMAP in Your Database Management System? (Pavlo et al., CIDR 2022)][pavlo]

[norvig]: https://colin-scott.github.io/personal_website/research/interactive_latency.html
[azure-disk]: https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types
[pavlo]: https://db.cs.cmu.edu/papers/2022/cidr2022-p13-crotty.pdf

## Implications for vmemkv

The crossover point sits around **50–100 µs** (the upper end of direct-attached enterprise NVMe).

| Storage speed                      | pread miss vs page fault | vmemkv competitiveness |
| ---------------------------------- | ------------------------ | ---------------------- |
| < 5 µs (latest NVMe)               | pread ≪ page fault       | worse                  |
| ~100–200 µs (KVM / enterprise NVMe)| pread ≈ page fault       | on par or better       |
| ~1–3 ms (cloud EBS)                | pread ≫ page fault       | advantageous           |
| ~5–15 ms (HDD)                     | pread ≫ page fault       | strongly advantageous  |
