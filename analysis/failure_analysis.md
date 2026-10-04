# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyen Manh Cuong  
**Khóa:** K4 - Track 3A  
**MSSV:** 2A202602650  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9000 | 0.9000 | +0.0000 |
| Answer Relevancy | 0.8500 | 0.8500 | +0.0000 |
| Context Precision | 0.8800 | 0.8800 | +0.0000 |
| Context Recall | 0.9274 | 0.8845 | -0.0429 |

---

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Trích từ `nghi_phep_nam_v2024.md`: Mỗi nhân viên chính thức được hưởng 15 ngày phép năm có lương, tăng từ 12 ngày so với chính sách năm 2023.
- **Worst metric:** `context_recall` (0.6760)
- **Error Tree:** Output có số ngày hiện tại nhưng thiếu đoạn văn bản gốc đối chiếu của quy chế cũ → Context chỉ lấy từ `nghi_phep_nam_v2024.md`, không lấy chunk từ `nghi_phep_nam_v2023.md` → Query tổng quát không chứa nhãn năm/phiên bản.
- **Root cause:** Xung đột phiên bản tài liệu (Document Version Conflict). Reranker xếp văn bản mới nhất lên đầu và top-3 bị chiếm bởi tài liệu v2024, làm mất tài liệu lịch sử v2023 cần cho đối chiếu trọn vẹn.
- **Suggested fix:** Bổ sung metadata filter cho phiên bản tài liệu (active vs deprecated) hoặc triển khai Temporal Query Rewriting để truy xuất thêm bối cảnh lịch sử quy định khi có thay đổi chính sách.

---

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trích từ `nghi_phep_nam_v2024.md` trả lời được 18 ngày phép (15 + 3), nhưng thiếu hoàn toàn dải lương Senior từ file `bang_luong_2024.md`.
- **Worst metric:** `context_recall` (0.6783)
- **Error Tree:** Output trả lời thiếu một nửa câu hỏi (khung lương Senior) → Context trích xuất chỉ chứa văn bản nghỉ phép, không có bảng lương → Query phức hợp gồm 2 câu hỏi con độc lập trong cùng một prompt.
- **Root cause:** Vấn đề Multi-hop / Multi-aspect Query. Khi query chứa cả ý về thâm niên ngày phép lẫn thang bảng lương, Bi-Encoder và Cross-Encoder bị thiên lệch (bias) về các chunk có mật độ từ khóa cao hơn (nghỉ phép/thâm niên), đẩy các chunk về bảng lương ra khỏi top-k.
- **Suggested fix:** Áp dụng kỹ thuật Sub-query Decomposition (phân rã câu hỏi thành 2 truy vấn riêng biệt: "nhân viên 9 năm thâm niên được bao nhiêu ngày phép" và "lương nhân viên Senior") sau đó hợp nhất kết quả tìm kiếm.

---

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Trích từ `mua_sam.md`: Mua sắm thiết bị CNTT cần xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất, đơn hàng khẩn cấp cần giải trình bằng văn bản.
- **Worst metric:** `context_recall` (0.7000)
- **Error Tree:** Output trả về quy định kỹ thuật thiết bị CNTT nhưng không đề cập thẩm quyền ký duyệt CEO cho mức 55 triệu → Context trích xuất đoạn danh mục kỹ thuật thay vì đoạn bảng phân quyền phê duyệt hạn mức.
- **Root cause:** Từ khóa "thiết bị" kéo chunk về quy chuẩn thiết bị CNTT lên vị trí cao hơn chunk chứa bảng ma trận thẩm quyền phê duyệt hạn mức tài chính (> 50 triệu).
- **Suggested fix:** Cải tiến BM25 tokenization và Dense Search với Metadata Enrichment (gắn tag `category: financial_approval` cho các điều khoản phân quyền) để lọc đúng chunk thẩm quyền ngân sách.

---

### #4
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Trích từ `ky_luong.md`: Phiếu lương điện tử được gửi qua email; thông tin lương là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp.
- **Worst metric:** `context_recall` (0.7526)
- **Error Tree:** Output trả lời đúng "Bí mật" nhưng thiếu định nghĩa kỹ thuật "Cấp 3" và quy định bảo mật kèm theo → Context chỉ chứa `ky_luong.md`, thiếu `phan_loai_du_lieu.md`.
- **Root cause:** Cross-Document Reference. Khái niệm phân loại dữ liệu nằm ở tài liệu an toàn thông tin, trong khi quy định lương nằm ở tài liệu nhân sự. RAG chưa liên kết được các tài liệu mang tính tham chiếu chéo.
- **Suggested fix:** Bổ sung Contextual Enrichment ở Module 5 để trích xuất các liên kết thực thể (entity linking) và dẫn chiếu chéo giữa các văn bản quy định.

---

### #5
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Trích từ `nghi_phep_khong_luong.md`: Nghỉ từ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO).
- **Worst metric:** `context_recall` (0.7893)
- **Error Tree:** Output đúng cấp phê duyệt CEO nhưng thiếu điều kiện phụ về đóng bảo hiểm xã hội → Chunk lấy được phần bảng ngày nhưng bị ngắt trước phần ghi chú ảnh hưởng phúc lợi.
- **Root cause:** Cắt đoạn con (Child chunking) theo độ dài 256 ký tự làm chia cắt điều khoản chính và điều khoản ràng buộc nghĩa vụ bảo hiểm nằm ngay phía sau.
- **Suggested fix:** Tận dụng triệt để Hierarchical Chunking (khi tìm kiếm bằng Child chunk nhưng trả về Parent chunk hoàn chỉnh lên tới 2048 ký tự) để đảm bảo không bị mất các điều khoản ghi chú bổ sung.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Không hoàn toàn: Trả lời đúng phần thâm niên phép năm (18 ngày) nhưng thiếu hoàn toàn phần khung lương Senior (20 - 35 triệu VNĐ/tháng).
2. **Context đúng?** → Sai/Thiếu: Context đưa vào LLM chỉ có các đoạn văn từ `nghi_phep_nam_v2024.md`, hoàn toàn vắng mặt văn bản `bang_luong_2024.md`.
3. **Query rewrite OK?** → Chưa có: Hệ thống gửi thẳng câu hỏi ghép vào Bi-Encoder/BM25 dẫn đến từ khóa "thâm niên", "nghỉ phép" lấn át từ khóa "lương Senior".
4. **Fix ở bước:**  
   - Bổ sung tầng **Query Decomposition** ở tiền xử lý: Tách thành Query A ("Nhân viên thâm niên 9 năm được bao nhiêu ngày phép?") và Query B ("Mức lương cấp bậc Senior là bao nhiêu?").
   - Sau đó chạy Hybrid Search song song cho 2 Query rồi gộp tập Context bằng RRF trước khi đưa vào Cross-Encoder Reranker.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Triển khai **Multi-Query Decomposition** với LangChain/LiteLLM để phân rã tự động các câu hỏi kép phức tạp.
- Tinh chỉnh **Metadata Filtering** theo phiên bản hiệu lực (`valid_date`, `status: active`) để triệt tiêu hoàn toàn xung đột giữa văn bản cũ và văn bản mới.
