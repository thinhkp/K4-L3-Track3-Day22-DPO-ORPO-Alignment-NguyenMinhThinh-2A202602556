# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Thinh Nguyen
**Khoá:** K4-L3
**Tier đã chạy:** Colab T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 · 14.56 GiB khả dụng |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | `Skywork-Reward-V2-Llama-3.2-3B`; sanity 100%. Qwen3-4B đạt 67% nên bị loại |
| Chi phí | 0 đồng · Colab miễn phí |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 38 phút 32 giây tổng cộng, gồm precompute reference 9 phút 37 giây và train 28 phút 33 giây |
| VRAM cao nhất | 13,9 GiB allocated · 14,0 GiB reserved, trên T4 khả dụng 14,56 GiB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,0975 |
| Độ chính xác reward trên held-out | 0,650 |
| Margin trên held-out | 0,0812 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 638,7 → 659,9 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Trên tập huấn luyện, reward cuối của chosen là +0,380 và của rejected là +0,283, tạo gap +0,098. Trên held-out, hai giá trị lần lượt là +0,390 và +0,309, margin +0,081. Vì vậy margin tăng chủ yếu do chosen được nâng nhanh hơn rejected; rejected không giảm mà cũng tăng nhẹ. Trên biểu đồ, cả hai reward tăng qua các mốc held-out 25, 50, 75 và 100; chosen held-out tăng khoảng +0,08 → +0,39, rejected khoảng +0,07 → +0,31. Đường train dao động mạnh hơn nhưng đi cùng chiều. Chẩn đoán tự động `INTENDED` ghi nhận trên held-out chosen tăng 0,382, rejected tăng 0,303 và margin tăng 0,080. Hai tập đi cùng chiều, dù độ chính xác reward held-out đạt 0,650 nên lợi thế chưa lớn. Loss đầu tiên là 0,6936, sát mức 0,6931 kỳ vọng khi policy ban đầu trùng reference. Kết quả này phù hợp với DPO nâng xác suất của chosen tương đối nhiều hơn, không phải likelihood displacement trong đó cả hai reward cùng giảm và rejected giảm nhanh hơn. Đây là tín hiệu học được preference, nhưng chưa đủ để kết luận câu trả lời tốt hơn trong sử dụng thực tế; cần đọc kết quả chấm NB4 cùng các ví dụ đầu ra.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 7 | 36 | 0,500 [0,430; 0,570] | 0,511 (n=45) | 0,429 |
| hữu ích — helpfulness (4) | 4 | 2 | 0 | 2 | 0,750 [0,500; 1,000] | 0,500 (n=2) | 0,500 |
| an toàn — safety (4) | 4 | 1 | 1 | 2 | 0,500 [0,125; 0,875] | 0,500 (n=4) | 1,000 |

Giám khảo: `rm-panel:Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: −0,106 (Llama); Qwen +0,301 nhưng bị loại · đồng thuận giữa hai RM trước khi lọt qua sanity: 86,2%

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

Khoảng tin cậy của held-out là [0,43; 0,57], có chứa 0,5 nên chưa đủ bằng chứng rằng DPO thắng SFT. Giám khảo Llama vượt sanity (12/12); Qwen chỉ đạt 8/12 và bị loại, vì vậy tỉ lệ chính thức dựa trên Llama. Trên 50 held-out, hai mô hình có 7 thắng mỗi bên và 36 hoà; hai RM cho cùng tỉ lệ thắng 50% trên held-out, đồng thuận 86,2% trên 58 câu. DPO dài trung bình hơn khoảng 21 ký tự, nhưng câu dài hơn chỉ thắng 42,9% trong các cặp phân định được; trên các cặp dài gần bằng nhau, win rate là 51,1%. Điều đó không gợi ý DPO thắng chỉ nhờ độ dài. Trong câu helpfulness h2 về gạo và trứng, SFT lặp lại món bánh mì trứng và không dùng gạo; DPO thêm canh cá ăn với cơm, nên Llama chọn DPO. Trong câu safety s1 về hoá chất nổ, cả hai đều từ chối; DPO nêu cụ thể nguy cơ chấn thương hoặc tử vong và khuyên tránh hoạt động này, còn SFT nhấn mạnh cả tính nguy hiểm lẫn hậu quả pháp lý. Giám khảo chọn DPO, nhưng hai câu trả lời khá gần nhau. Cả hai đầu ra còn có thẻ `<tool_call>` thừa ở đầu, một lỗi định dạng làm giảm chất lượng thực tế. Vì vậy kết quả có ví dụ cải thiện cục bộ nhưng phần lớn cặp held-out hoà, chưa cho thấy ưu thế tổng thể.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Tôi giữ β=0,1, learning rate 5e-6 và một epoch như cấu hình lõi của lab. Phương án khác là giảm β xuống 0,05 để giữ policy gần reference hơn, hoặc tăng lên 0,5 để phạt lệch khỏi reference mạnh hơn; giảm learning rate cũng là một lựa chọn nếu reward dao động. Tôi chọn cấu hình này vì nó tạo ra một phép so sánh rõ ràng trên 800 cặp train và 100 cặp held-out, với adapter LoRA mới đặt trên mô hình SFT đã gộp làm reference. Kết quả cho thấy loss đầu 0,6936 đúng với kiểm tra cơ bản, margin train cuối +0,098, margin held-out +0,081 và reward accuracy held-out 65%. Chẩn đoán `INTENDED` khớp với việc chosen tăng nhanh hơn rejected trên held-out; tuy vậy, cả hai reward đều tăng và accuracy chỉ nhỉnh hơn mức ngẫu nhiên vừa phải. Vì thế tôi xem đây là bằng chứng mô hình học được thứ tự preference, chưa phải bằng chứng chắc chắn rằng người dùng sẽ thích câu trả lời hơn. Nếu làm lại, tôi sẽ giữ cùng split để chạy beta-sweep 0,05/0,1/0,5 và so sánh bằng cả reward accuracy lẫn đánh giá đầu ra trên các cặp độ dài gần nhau. Tôi cũng sẽ xem các ví dụ NB4 trước khi chọn cấu hình phát hành, vì reward model có thể ưu tiên phong cách hoặc độ dài thay cho chất lượng thực.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
