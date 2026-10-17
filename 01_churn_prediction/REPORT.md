# Dự án 1 — Dự đoán khách hàng rời mạng (Telco churn) và tối ưu chiến dịch giữ chân

**Dữ liệu:** Telco Customer Churn (IBM, công khai) — 7.043 khách hàng, 20 biến, tỷ lệ churn **26,5%**.
Script tự tải dữ liệu thật; nếu không có mạng thì sinh dữ liệu mô phỏng cùng schema để pipeline vẫn chạy được.

**Chạy:** `python 01_churn_prediction/src/train.py` (~16 giây)

---

## 1. Bài toán được đặt theo góc nhìn kinh doanh

Không dừng ở "dự đoán ai sẽ rời mạng" mà trả lời: **nên gọi giữ chân ai để lãi cao nhất?**

Giả định chi phí (ghi rõ trong code, dễ thay đổi):

| Tham số | Giá trị |
|---|---|
| Chi phí ưu đãi / khách được liên hệ | 60 USD |
| Tỷ lệ ưu đãi giữ được khách thật sự định rời | 30% |
| Biên lợi nhuận gộp | 35% |
| Khoảng thời gian tính giá trị | 12 tháng |

## 2. EDA — những gì dữ liệu nói ngay

| Nhóm | Tỷ lệ churn |
|---|---|
| Hợp đồng tháng | **42,7%** |
| Hợp đồng 1 năm | 11,3% |
| Hợp đồng 2 năm | 2,8% |
| Internet Fiber optic | **41,9%** |
| Internet DSL | 19,0% |
| Không dùng internet | 7,4% |

→ Churn tập trung ở nhóm hợp đồng tháng + fiber. Đây là giả thuyết dẫn đường cho feature engineering.

## 3. Feature engineering

- `n_addons` — số dịch vụ cộng thêm đang dùng (đại diện cho độ "dính")
- `is_autopay` — thanh toán tự động (đòn bẩy giữ chân đã biết)
- `tenure_bucket` — giai đoạn vòng đời thay cho số tháng thô
- `arpu_hist = TotalCharges / tenure` và `price_delta = MonthlyCharges − arpu_hist` — phát hiện khách vừa bị tăng giá/upsell
- `charges_per_addon` — mức chi trên mỗi dịch vụ

Toàn bộ xử lý nằm trong `Pipeline` + `ColumnTransformer` nên không rò rỉ dữ liệu khi cross-validation.

## 4. So sánh mô hình (5-fold stratified CV trên tập train)

| Mô hình | ROC-AUC | PR-AUC |
|---|---|---|
| **Logistic Regression** | **0,847 ± 0,011** | 0,661 |
| Random Forest | 0,845 ± 0,009 | 0,659 |
| HistGradientBoosting | 0,845 ± 0,010 | 0,660 |

Ba mô hình chênh nhau trong khoảng sai số → **chọn hồi quy logistic** vì đơn giản, giải thích được và nhanh. Đây là kết luận có giá trị: dữ liệu dạng bảng nhỏ không tự động cần boosting.

## 5. Hiệu chỉnh xác suất (calibration)

Quyết định chi tiền dựa trên xác suất, nên xác suất phải *đúng*, không chỉ *xếp hạng đúng*:

| | Brier score |
|---|---|
| Chưa hiệu chỉnh | 0,1656 |
| Isotonic calibration | **0,1358** (−18%) |

## 6. Kết quả trên tập test độc lập (1.409 khách)

| Chỉ số | Giá trị |
|---|---|
| ROC-AUC | **0,845** |
| PR-AUC | 0,651 |
| Lift ở decile cao nhất | **2,81×** |
| Bắt được churn khi liên hệ 30% danh sách | **65%** |
| Ngưỡng tối ưu theo lợi nhuận | 0,705 |
| Precision / Recall tại ngưỡng đó | 0,80 / 0,24 |

## 7. Điểm quan trọng nhất: ngưỡng tối ưu ≠ 0,5

| Chiến lược | Giá trị ròng trên 1.409 khách test |
|---|---|
| Liên hệ tất cả | **−50.249 USD** |
| Liên hệ theo ngưỡng tối ưu (113 khách) | **+2.293 USD** |

Quy đổi: **≈ +1.628 USD trên mỗi 1.000 khách hàng**.

Vì lợi nhuận kỳ vọng khi giữ được một khách (~88 USD) không lớn hơn nhiều so với chi phí ưu đãi (60 USD), mô hình chỉ nên liên hệ nhóm xác suất rất cao → ngưỡng 0,705 chứ không phải 0,5. Nếu tối ưu theo F1 thì ngưỡng là 0,31 và chiến dịch **lỗ** — một minh chứng cho việc chọn chỉ số theo bài toán kinh doanh.

## 8. Yếu tố ảnh hưởng (permutation importance, mức giảm ROC-AUC)

| Biến | Mức giảm AUC |
|---|---|
| tenure | 0,050 |
| Contract | 0,044 |
| InternetService | 0,025 |
| tenure_bucket | 0,016 |
| OnlineSecurity | 0,004 |
| TechSupport | 0,003 |

→ Khuyến nghị: đẩy khách hợp đồng tháng mới (0–6 tháng) sang hợp đồng dài hạn, kèm gói TechSupport/OnlineSecurity và thanh toán tự động.

## 9. Kết quả xuất ra

- `reports/metrics.json` — toàn bộ số liệu
- `reports/scored_customers.csv` — danh sách khách đã chấm điểm, cột `target_for_campaign` dùng được ngay
- `reports/churn_model.joblib` — model đã huấn luyện và hiệu chỉnh
- `reports/figures/` — 5 hình: EDA theo phân khúc, tenure/giá, ROC-PR-calibration, đường lợi nhuận theo ngưỡng + cumulative gain, feature importance

## 10. Hạn chế (nói thẳng khi phỏng vấn)

- Dữ liệu là ảnh chụp một thời điểm, không có mốc thời gian → không mô hình hoá được churn theo thời gian (survival analysis sẽ phù hợp hơn).
- Các tham số chi phí là giả định; cần số thật từ phòng marketing.
- Mô hình dự đoán *tương quan*, không phải *nhân quả*: chưa trả lời được "ưu đãi làm tăng tỷ lệ giữ khách bao nhiêu". Việc đó cần một thử nghiệm A/B (xem dự án 4) hoặc uplift modeling.
