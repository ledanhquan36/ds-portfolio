# Dự án 2 — Dự báo nhu cầu theo giờ, trước 24 giờ (time series)

**Dữ liệu:** UCI Bike Sharing (2011–2012) lấy qua OpenML — **17.379 quan sát theo giờ**, kèm thời tiết, ngày lễ, mùa.
**Chạy:** `python 02_demand_forecasting/src/forecast.py` (~33 giây)

---

## 1. Bài toán

Mỗi 00:00 phải công bố kế hoạch nhân sự/điều phối xe cho 24 giờ tiếp theo.
→ Mọi đặc trưng chỉ được dùng thông tin có **trước thời điểm dự báo 24 giờ**: lag nhỏ nhất là 24h, mọi rolling window đều `shift(24)`.
Đây là điểm người mới hay làm sai (dùng `lag_1h` cho dự báo ngày hôm sau) và là lỗi rò rỉ dữ liệu kinh điển.

## 2. Xử lý dữ liệu — phần kỹ thuật đáng kể nhất

Bản OpenML giữ thứ tự thời gian nhưng **đã mất cột ngày**. Script dựng lại trục thời gian:

1. Tách các ngày bằng điểm "giờ quay vòng" (hour giảm so với dòng trước).
2. Ghép từng nhóm ngày vào lịch thật 2011–2012 theo bộ ba `(year, month, weekday)` có sẵn trong dữ liệu.
3. **Kiểm tra bắt buộc**: nếu weekday/month dựng lại không khớp 100% hoặc có timestamp trùng → raise `AssertionError`.

Lần chạy đầu tiên check này **đã bắt được lỗi thật**: ngày 29/10/2012 (bão Sandy) hệ thống ngừng giữa trưa và dòng tiếp theo đã sang ngày 30 mà giờ vẫn tăng, nên quy tắc "giờ quay vòng" bỏ sót một ngày. Sau khi bổ sung điều kiện đổi `weekday`, kết quả: **weekday khớp 1.000, month khớp 1.000, 0 timestamp trùng**, 165 giờ thiếu so với lưới đầy đủ (đúng như mô tả của bộ dữ liệu gốc).

## 3. Đặc trưng

- Lịch: giờ, thứ, tháng, năm, cuối tuần, ngày trong năm + mã hoá tuần hoàn `sin/cos` (giờ và năm)
- Ngoại sinh biết trước: ngày lễ, ngày làm việc, thời tiết, nhiệt độ, độ ẩm, gió
- Trễ: `lag 24/25/26/48/72/168/336h`
- Rolling (đã shift 24h): mean/std/max 24h, mean 168h
- `same_hour_3w_mean` — trung bình cùng giờ 3 tuần trước

## 4. Kiểm định đúng chuẩn chuỗi thời gian

- Chọn mô hình: `TimeSeriesSplit` 5 fold mở rộng dần, mỗi fold validation 14 ngày
- Đánh giá cuối: **8 tuần cuối (06/11–31/12/2012) hoàn toàn không tham gia chọn mô hình**
- Train: 15.072 giờ | Test: 1.267 giờ

## 5. Baseline — so với cách phòng vận hành thật sự làm

| Baseline | MAE | RMSE | sMAPE |
|---|---|---|---|
| Cùng giờ hôm qua | 63,4 | 106,7 | 46,2% |
| Cùng giờ tuần trước (seasonal naive) | 72,6 | 119,9 | 45,9% |
| Trung bình cùng giờ 3 tuần trước | 66,8 | 112,7 | 41,0% |

## 6. Kết quả mô hình

Cross-validation (MAE):

| Mô hình | CV MAE |
|---|---|
| Ridge (one-hot + cyclical) | 50,3 ± 5,9 |
| HistGradientBoosting (squared) | 40,0 ± 3,1 |
| **HistGradientBoosting (Poisson)** | **37,9 ± 4,0** |

Trên 8 tuần test:

| Mô hình | MAE | RMSE | sMAPE | R² |
|---|---|---|---|---|
| Ridge | 51,7 | 84,1 | 47,1% | 0,762 |
| HGB squared | 40,7 | 65,5 | 37,5% | 0,856 |
| **HGB Poisson** | **38,7** | **61,2** | **28,9%** | **0,874** |

→ **Giảm 46,7% MAE so với seasonal naive** (38,7 vs 72,6).
Dùng **Poisson loss** thay vì bình phương sai số vì biến mục tiêu là số đếm không âm: tốt hơn 2 điểm MAE và 8,6 điểm sMAPE so với cùng mô hình với squared loss.

## 7. Khoảng dự báo và một phát hiện trung thực

Quantile gradient boosting (5%–95%) cho khoảng danh nghĩa 90%, nhưng **độ phủ thực tế chỉ 79,6%**, độ rộng trung bình 142 chuyến/giờ.

Khoảng dự báo bị hụt phủ trên dữ liệu tương lai — vì phân phối tháng 11–12 (lạnh, lễ) khác tập train. Nếu dùng để lập kế hoạch nhân sự thì phải hiệu chuẩn lại (conformal prediction) chứ không tin con số 90% danh nghĩa. **Nêu đúng hạn chế này quan trọng hơn là báo một con số đẹp.**

## 8. Phân tích sai số

- `same_hour_3w_mean` là đặc trưng mạnh nhất (bỏ ra → MAE tăng 75,7), sau đó là `dayofyear`, `workingday`, `hour_cos`, `lag_24h`, `humidity`
- Sai số lớn nhất ở các giờ cao điểm 8h và 17–18h — đúng như kỳ vọng vì phương sai nhu cầu ở các giờ đó lớn nhất
- Phân phối phần dư gần đối xứng quanh 0, không có độ chệch hệ thống
- Nhìn hình `01_forecast_vs_actual.png`: tuần Giáng sinh (22–26/12) mô hình **dự báo cao hơn thực tế rõ rệt** vì chỉ có 1 mùa lễ trong tập train để học. Đây là lỗi dạng "chế độ hiếm" — cách xử lý thực tế là thêm cờ ngày lễ/kỳ nghỉ chi tiết hoặc hạ trọng số dự báo trong tuần lễ.

## 9. Kết quả xuất ra

`reports/metrics.json`, `reports/forecast_test.csv` (thực tế / dự báo / khoảng tin cậy từng giờ), `reports/cv_results.csv`, 5 hình trong `reports/figures/`.

## 10. Hạn chế

- Thời tiết trong tập dữ liệu là **giá trị đã xảy ra**, thực tế chỉ có *dự báo* thời tiết → sai số thật sẽ cao hơn một chút.
- Chỉ 2 năm dữ liệu nên mùa vụ năm chỉ có 2 chu kỳ.
- Chưa thử SARIMAX/Prophet; với dữ liệu nhiều đặc trưng ngoại sinh thì cách tiếp cận dạng bảng + GBM thường thắng, nhưng nên kiểm chứng.
