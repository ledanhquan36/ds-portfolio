# Data Science Portfolio — 4 dự án end-to-end

Bốn dự án độc lập, mỗi dự án chạy bằng **một câu lệnh**, tự tải dữ liệu, tự xuất chỉ số và hình ảnh.
Ba dự án dùng **dữ liệu công khai thật**; dự án thống kê dùng mô phỏng *có chủ đích* vì cần biết ground truth để chứng minh phương pháp đúng.

```bash
pip install -r requirements.txt
python run_all.py          # chạy cả 4, tổng ~2 phút trên CPU
```

| # | Dự án | Kỹ năng chính | Dữ liệu | Kết quả nổi bật |
|---|---|---|---|---|
| 1 | [Dự đoán churn + tối ưu chiến dịch giữ chân](01_churn_prediction/REPORT.md) | ML phân loại, calibration, tối ưu theo lợi nhuận | Telco Churn (IBM), 7.043 KH | ROC-AUC **0,845**; nhắm mục tiêu có lãi **+1.628 USD/1.000 KH** trong khi gọi toàn bộ danh sách thì **lỗ 50.249 USD** |
| 2 | [Dự báo nhu cầu theo giờ, trước 24h](02_demand_forecasting/REPORT.md) | Time series, feature trễ, validation theo thời gian | UCI Bike Sharing, 17.379 giờ | MAE **38,7** vs seasonal naive 72,6 → **tốt hơn 46,7%**; R² 0,874 |
| 3 | [Phân loại văn bản 20 lớp + topic modeling](03_review_nlp/REPORT.md) | NLP, TF-IDF, NMF, phát hiện rò rỉ dữ liệu | 20 Newsgroups, 18.846 văn bản | Macro-F1 **0,697**; chứng minh metadata thổi phồng benchmark **+0,175 F1** |
| 4 | [Phân tích thử nghiệm A/B](04_ab_testing/REPORT.md) | Thống kê suy diễn, power, CUPED, Bayesian | Mô phỏng có ground truth | Phát hiện lift thật +6% (CI [+5,0%; +11,3%]); peeking làm sai số loại I tăng **5% → 16,3%** |

---

## Mỗi dự án có gì

```
0X_ten_du_an/
├── REPORT.md              # báo cáo đầy đủ: bài toán, phương pháp, số liệu thật, hạn chế
├── src/                   # code chạy được, có docstring giải thích lựa chọn
├── data/                  # dữ liệu tự tải về (đã .gitignore)
└── reports/
    ├── metrics.json       # toàn bộ chỉ số, tái lập được
    ├── *.csv              # kết quả dùng được ngay (danh sách KH đã chấm điểm, dự báo từng giờ...)
    └── figures/*.png      # hình đã xuất sẵn
```

## Nguyên tắc kỹ thuật áp dụng xuyên suốt

1. **Không rò rỉ dữ liệu.** Mọi tiền xử lý nằm trong `sklearn.Pipeline`; chuỗi thời gian chỉ dùng thông tin có trước thời điểm dự báo; thí nghiệm ở dự án 3 **đo** mức rò rỉ thay vì nói suông.
2. **Baseline trước mô hình.** Dự án 2 so với đúng cách mà phòng vận hành đang làm (cùng giờ hôm qua / tuần trước), không so với baseline rơm.
3. **Chỉ số phải khớp quyết định.** Dự án 1 tối ưu ngưỡng theo giá trị kỳ vọng của chiến dịch, không theo accuracy/F1 — và chỉ ra rằng tối ưu F1 sẽ làm chiến dịch lỗ.
4. **Xác suất phải hiệu chỉnh.** Isotonic calibration giảm Brier score 18%, vì quyết định chi tiền phụ thuộc vào xác suất tuyệt đối.
5. **Kiểm tra dữ liệu bằng assertion.** Dự án 2 dựng lại trục thời gian bị mất rồi raise lỗi nếu không khớp 100% — và cách này đã bắt được một lỗi thật (ngày bão Sandy 29/10/2012).
6. **Nói rõ hạn chế.** Mỗi REPORT.md có mục hạn chế: khoảng dự báo chỉ phủ 79,6% so với 90% danh nghĩa, giả định chi phí, nhãn chồng lấn, mô hình không suy diễn nhân quả.

## Môi trường

Python 3.10, chỉ dùng numpy / pandas / scikit-learn / scipy / statsmodels / matplotlib. Không cần GPU, không cần tài khoản Kaggle.

---

## Gạch đầu dòng gợi ý cho CV

Chọn 2–3 dòng, đừng đưa cả bốn. Mỗi dòng theo cấu trúc *việc đã làm → cách làm → con số*:

**Dự án 1 — churn**
> Xây pipeline dự đoán khách hàng rời mạng trên 7.043 khách hàng (scikit-learn): feature engineering theo nghiệp vụ, so sánh 3 mô hình bằng stratified CV, hiệu chỉnh xác suất bằng isotonic (Brier −18%), đạt ROC-AUC 0,845 và lift 2,8× ở decile cao nhất. Tối ưu ngưỡng theo giá trị kỳ vọng của chiến dịch thay vì F1, chuyển một chiến dịch lỗ 50.000 USD thành có lãi ~1.600 USD/1.000 khách hàng.

**Dự án 2 — dự báo**
> Dự báo nhu cầu theo giờ trước 24 giờ trên 17.379 quan sát: đặc trưng trễ/rolling tuân thủ ràng buộc thời gian thực, validation bằng expanding-window TimeSeriesSplit và 8 tuần hold-out, gradient boosting với Poisson loss đạt MAE 38,7 — tốt hơn baseline nghiệp vụ 46,7%; kèm khoảng dự báo bằng quantile regression và đánh giá độ phủ thực tế (79,6% so với 90% danh nghĩa).

**Dự án 3 — NLP**
> Phân loại văn bản 20 lớp (18.846 tài liệu) với TF-IDF + ComplementNB/LinearSVC đạt macro-F1 0,697; thiết kế thí nghiệm chứng minh metadata của bộ benchmark thổi phồng macro-F1 thêm 0,175, và dùng NMF để trích 15 chủ đề không cần nhãn.

**Dự án 4 — thống kê/experimentation**
> Xây bộ phân tích A/B test: power analysis & MDE, kiểm tra sample ratio mismatch, z-test/bootstrap/delta method cho chỉ số tỷ lệ, CUPED giảm 9,4% phương sai (tương đương 1,1× lưu lượng), mô phỏng định lượng tác hại của việc "peeking" (sai số loại I 5% → 16,3%) và hiệu chuẩn ngưỡng chuỗi, cùng quyết định Bayesian theo ngưỡng đăng ký trước.

### Dòng mô tả mục Projects (ngắn gọn)
> **Data Science Portfolio** — 4 dự án end-to-end (churn & tối ưu chiến dịch, dự báo chuỗi thời gian, NLP phân loại văn bản, phân tích A/B test) trên dữ liệu công khai; Python, scikit-learn, pandas, scipy. `github.com/ledanhquan36/ds-portfolio`



```bash
git init && git add . && git commit -m "Data science portfolio: 4 end-to-end projects"
git remote add origin https://github.com/ledanhquan36/ds-portfolio.git
git push -u origin main
```

