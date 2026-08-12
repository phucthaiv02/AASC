# Attack variants

Mỗi biến thể là một file Python độc lập định nghĩa `AttackAlgorithm` theo contract của
`aicomp_sdk`.

Quy ước tên gợi ý:

```text
attacks/
  live_fill_m47.py
  live_fill_m42.py
  replay_safe_hops1.py
```

Khi chạy, code tự tạo thư mục
`artifacts/runs/<tên-file>_<YYYYMMDD_HHMMSS_microseconds>` để giữ lịch sử và tạo leaderboard:

```bash
aas-nim validate \
  --attack attacks/live_fill_m47.py \
  --budget-s 300
```

Không import biến thể này từ biến thể khác. Kaggle submission cần một file tự chứa; khi chọn
được bản tốt nhất, copy nội dung của nó sang `attack.py` để submit.
