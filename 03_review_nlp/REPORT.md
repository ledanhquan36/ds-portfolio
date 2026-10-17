# Dự án 3 — Phân loại văn bản 20 lớp + khai phá chủ đề, và thí nghiệm về rò rỉ dữ liệu

**Dữ liệu:** 20 Newsgroups (split train/test chính thức) — 11.314 văn bản train, 7.532 test, **20 lớp**.
**Chạy:** `python 03_review_nlp/src/text_classification.py` (~50 giây)

---

## 1. Điểm khác biệt của dự án này

Benchmark 20 Newsgroups mà hầu hết blog/khoá học dùng **để nguyên email header, footer và phần quote lại**. Những phần đó chứa tên nhóm, địa chỉ email, chữ ký — tức là **rò rỉ nhãn**. Dự án này huấn luyện **cùng một pipeline hai lần** để đo chính xác mức rò rỉ đó.

| Cấu hình (cùng pipeline TF-IDF + LinearSVC) | Accuracy | Macro-F1 | Số đặc trưng TF-IDF |
|---|---|---|---|
| Giữ nguyên metadata (cách làm phổ biến) | 0,8634 | **0,8577** | 131.711 |
| Bỏ header/footer/quote (trung thực) | 0,6964 | **0,6831** | 56.618 |

→ **Metadata thổi phồng macro-F1 lên 0,175 tuyệt đối (+25,6% tương đối).**
Mọi kết quả sau đây đều báo trên bản *đã làm sạch*.

## 2. Pipeline

TF-IDF: chữ thường, bỏ dấu, stop words tiếng Anh, token chỉ gồm chữ cái (loại số), n-gram 1–2, `min_df=3`, `max_df=0.6`, `sublinear_tf=True`.

## 3. So sánh mô hình (trên tập test chính thức, văn bản đã làm sạch)

| Mô hình | Accuracy | Macro-F1 | Thời gian fit |
|---|---|---|---|
| **ComplementNB** (α=0,3) | **0,7136** | **0,6966** | 3,5s |
| LinearSVC (C=0,5) | 0,6964 | 0,6831 | 4,0s |
| Logistic Regression (C=8) | 0,6925 | 0,6823 | 17,7s |

**ComplementNB thắng** — đúng như thiết kế của nó cho dữ liệu văn bản lệch lớp, và nhanh hơn logistic regression 5 lần. Một kết quả hữu ích: mô hình "cổ điển" vẫn là baseline rất mạnh cho phân loại văn bản chủ đề.

## 4. Lớp nào dễ, lớp nào khó

| Lớp tốt nhất | F1 | | Lớp kém nhất | F1 |
|---|---|---|---|---|
| rec.sport.hockey | 0,900 | | talk.religion.misc | **0,232** |
| rec.sport.baseball | 0,880 | | alt.atheism | 0,368 |
| talk.politics.mideast | 0,812 | | talk.politics.misc | 0,532 |
| comp.windows.x | 0,800 | | sci.electronics | 0,615 |
| sci.crypt | 0,798 | | comp.os.ms-windows.misc | 0,654 |

Các cặp nhầm nhiều nhất:

| Nhãn thật | Bị dự đoán thành | Số lần |
|---|---|---|
| talk.religion.misc | soc.religion.christian | 98 |
| talk.politics.misc | talk.politics.guns | 88 |
| alt.atheism | soc.religion.christian | 83 |
| comp.os.ms-windows.misc | comp.sys.ibm.pc.hardware | 40 |

→ Sai số **không phải lỗi mô hình mà do thiết kế nhãn**: ba nhóm tôn giáo và hai nhóm chính trị chồng lấn nội dung. Khuyến nghị thực tế: gộp nhãn hoặc chuyển sang gán nhiều nhãn (multi-label).

## 5. Learning curve

| Số văn bản train | Macro-F1 |
|---|---|
| 1.131 (10%) | 0,563 |
| 2.828 (25%) | 0,631 |
| 5.657 (50%) | 0,669 |
| 8.485 (75%) | 0,684 |
| 11.314 (100%) | 0,697 |

Đường cong đã gần phẳng: thêm dữ liệu cùng loại chỉ tăng ~1 điểm mỗi lần gấp đôi → muốn tiến xa hơn phải đổi **mô hình** (transformer) hoặc **nhãn**, không phải thêm dữ liệu.

## 6. Từ khoá quyết định (trọng số LinearSVC)

| Lớp | Từ có trọng số cao nhất |
|---|---|
| sci.space | space, orbit, launch, spacecraft, nasa, moon, shuttle, solar |
| rec.sport.hockey | hockey, nhl, puck, team, playoff, leafs, playoffs |
| talk.politics.guns | gun, guns, weapons, firearms, nra, weapon, firearm |
| comp.graphics | graphics, image, images, pov, animation, tiff, viewer |

Mô hình học đúng từ vựng chuyên ngành, không học "mẹo" — đây là kiểm tra bắt buộc trước khi tin một mô hình văn bản.

## 7. Khai phá chủ đề không dùng nhãn (NMF, 15 chủ đề)

Chủ đề tự tìm được khớp rất tốt với nhóm tin thật, ví dụ:

| Chủ đề | Từ khoá | Nhóm tin chiếm đa số |
|---|---|---|
| 5 | drive, hard, scsi, drives, disk, controller | comp.sys.ibm.pc.hardware |
| 7 | god, jesus, bible, believe, christians, christ | soc.religion.christian |
| 8 | key, chip, clipper, encryption, keys, escrow | sci.crypt |
| 6 | sale, offer, shipping, condition, price | misc.forsale |
| 3 | game, team, games, season, play, players | rec.sport.hockey |

Một vài chủ đề chỉ bắt được **chữ ký người dùng** (ví dụ "geb cadre dsl shameful surrender") — bằng chứng trực quan nữa cho việc metadata còn sót vẫn chi phối mô hình.

## 8. Kết quả xuất ra

`reports/metrics.json`, `reports/model_comparison.csv`, `reports/misclassified_examples.csv` (200 ca sai kèm trích đoạn văn bản), 4 hình: F1 từng lớp, confusion matrix 20×20, biểu đồ rò rỉ + learning curve, 15 chủ đề NMF.

## 9. Hạn chế

- Chưa thử transformer (DistilBERT fine-tune thường đạt macro-F1 ~0,75–0,80 ở cấu hình sạch này); dự án này cố ý giữ baseline nhẹ, chạy trên CPU trong 50 giây.
- Dữ liệu tiếng Anh, năm 1995; nếu áp dụng vào tiếng Việt cần tokenizer riêng (ví dụ `underthesea`, `pyvi`) vì khoảng trắng không phân tách từ trong tiếng Việt.
