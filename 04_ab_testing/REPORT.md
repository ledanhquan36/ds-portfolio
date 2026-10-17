# Dự án 4 — Bộ công cụ phân tích thử nghiệm A/B (thống kê suy diễn)

**Dữ liệu:** thử nghiệm checkout **mô phỏng có ground truth đã biết** (lift thật +6% tương đối).
Mô phỏng ở đây là *cố ý*: chỉ khi biết sự thật mới chứng minh được bộ công cụ cho ra kết luận đúng. Dữ liệu người dùng được xuất ra `data/ab_test_users.csv` để chạy lại.

**Chạy:** `python 04_ab_testing/src/ab_test_analysis.py` (~4 giây)

---

## 1. Thiết kế trước khi xem dữ liệu (pre-registration)

| Tham số | Giá trị |
|---|---|
| Conversion nền | 8,2% |
| α / power | 0,05 / 0,80 |
| Cỡ mẫu mỗi nhánh | 80.000 (tính từ power analysis, không đoán) |
| Ngưỡng quyết định đã đăng ký trước | chỉ ship nếu tin rằng lift > +2% tương đối |
| MDE tại cỡ mẫu này | **4,7% tương đối** |

Lần chạy đầu tôi đặt 12.000 người/nhánh → MDE 12,1%, lớn gấp đôi lift thật 6%, và thử nghiệm **không phát hiện được hiệu ứng có thật** (p = 0,63). Đó chính là lý do phải làm power analysis *trước*: một thử nghiệm thiếu lực kiểm định không cho kết luận "không có hiệu ứng", nó không cho kết luận gì cả.

## 2. Kiểm tra sức khoẻ trước khi đọc chỉ số (SRM)

80.000 vs 80.000, chi-square p = 1,000 → phân bổ lành mạnh. Nếu SRM thất bại thì **dừng**, mọi chỉ số phía sau đều không đáng tin.

## 3. Suy diễn trên các chỉ số

| Chỉ số | Control | Treatment | Hiệu ứng | p-value |
|---|---|---|---|---|
| **Conversion** (chính) | 9,008% | 9,744% | **+8,2% tương đối**, 95% CI [+5,0%; +11,3%] | < 0,0001 |
| Doanh thu / người (bootstrap 4.000 lần) | — | — | +0,74 USD, CI [+0,28; +1,22] | 0,0015 |
| Doanh thu / session (delta method) | — | — | +0,261 USD | 0,0024 |

Khoảng tin cậy **chứa đúng lift thật +6%** → bộ công cụ cho kết quả đúng.

Ba kỹ thuật khác nhau cho ba loại chỉ số:
- **z-test hai tỷ lệ** cho chỉ số nhị phân
- **bootstrap** cho doanh thu/người (phân phối lệch nặng, nhiều giá trị 0)
- **delta method** cho chỉ số tỷ lệ doanh thu/session, vì **mẫu số cũng là biến ngẫu nhiên** — dùng t-test thường ở đây sẽ ước lượng sai phương sai

## 4. CUPED — giảm phương sai bằng dữ liệu trước thử nghiệm

| | Giá trị |
|---|---|
| Tương quan chi tiêu trước/trong thử nghiệm | 0,306 |
| θ (hệ số điều chỉnh) | tính từ `cov(Y,X)/var(X)` |
| Giảm phương sai | **−9,4%** |
| Giảm sai số chuẩn | −4,8% |
| Tương đương lưu lượng | **1,10×** |

Nghĩa là: cùng một kết luận nhưng cần ít hơn ~10% traffic, tức rút ngắn thử nghiệm mà không mất tính hợp lệ.

## 5. "Peeking" — cái giá của việc xem kết quả mỗi ngày

Mô phỏng 3.000 thử nghiệm **A/A** (hai nhánh giống hệt nhau, không có hiệu ứng), kiểm tra 7 lần:

| Cách làm | Tỷ lệ dương tính giả |
|---|---|
| Chỉ xem 1 lần ở mốc đã định | **5,3%** (đúng như α = 5%) |
| Xem 7 lần, dùng ngưỡng 1,96 | **16,3%** |
| Xem 7 lần, dùng ngưỡng đã hiệu chuẩn | 5% |

Ngưỡng chuỗi (kiểu Pocock) được **hiệu chuẩn bằng mô phỏng**: `|z| > 2,44` thay vì 1,96. Đây là lý do một nửa "thắng" trong các công cụ A/B thương mại không tái lập được.

## 6. Nhiều chỉ số cùng lúc — Benjamini-Hochberg

| Chỉ số | p-value | Ngưỡng BH | Kết luận |
|---|---|---|---|
| conversion | < 0,0001 | 0,010 | có ý nghĩa |
| revenue_per_user | 0,0015 | 0,020 | có ý nghĩa |
| revenue_per_session | 0,0024 | 0,030 | có ý nghĩa |
| sessions_per_user | 0,3266 | 0,040 | không |
| pre_period_revenue_balance | 0,5758 | 0,050 | không (đúng — kiểm tra cân bằng) |

## 7. Quyết định theo ngôn ngữ kinh doanh (Bayesian)

Beta-Binomial, prior đều:

| | |
|---|---|
| P(treatment tốt hơn control) | **1,000** |
| P(lift > +2% đã đăng ký trước) | **1,000** |
| Khoảng tin cậy Bayes 95% của lift | khớp với CI tần suất |
| Tổn thất kỳ vọng nếu ship | ~0,000 điểm phần trăm |

**Kết luận theo quy tắc đã đăng ký trước: SHIP.** (Ground truth của mô phỏng: +6%.)

Cách báo cáo này tốt hơn p-value khi nói với stakeholder: "xác suất 100% rằng phiên bản mới tốt hơn, và nếu sai thì thiệt hại kỳ vọng gần như bằng 0" dễ hành động hơn "p < 0,05".

## 8. Kết quả xuất ra

`reports/metrics.json` (toàn bộ thiết kế, kiểm định, CUPED, peeking, BH, Bayes), `data/ab_test_users.csv`, 2 hình: (1) MDE + power curve + theo dõi z theo ngày với 2 ngưỡng, (2) độ rộng CI trước/sau CUPED + tỷ lệ dương tính giả + posterior.

## 9. Hạn chế

- Dữ liệu mô phỏng: chứng minh **tính đúng của phương pháp**, không phải một kết quả kinh doanh thật.
- Chưa xử lý: hiệu ứng mạng (network effect), nhiễu chéo giữa các nhánh, novelty effect dài hạn, và heterogeneous treatment effect (cần uplift/CATE modeling).
- Ngưỡng chuỗi hiệu chuẩn bằng mô phỏng chỉ đúng cho đúng lịch kiểm tra đó; alpha-spending (O'Brien-Fleming) tổng quát hơn.
