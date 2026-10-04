# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyen Manh Cuong  
**Khóa:** K4 - Track 3A  
**MSSV:** 2A202602650  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa các lý thuyết RAG nâng cao trong bài giảng và việc hiện thực hóa trong mã nguồn:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|----------------|--------|-------------|-------------------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Sử dụng mô hình `all-MiniLM-L6-v2` tính cosine similarity giữa các câu liên tiếp. Với ngưỡng `threshold=0.85` (hoặc `0.5` trong test), văn bản được chia theo chuyển dịch chủ đề tự nhiên thay vì chia thô theo số ký tự. Nhờ vậy không bị đứt câu hoặc tách rời bối cảnh lập luận. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Tạo cấu trúc phân cấp Parent (2048 ký tự) và Child (256 ký tự). Khi tìm kiếm, so khớp dựa trên Child chunk để đạt độ đặc hiệu (precision) cao nhất, nhưng khi nạp vào LLM thì sử dụng `parent_id` để trả về Parent chunk bao hàm đầy đủ bối cảnh xung quanh. |
| **Structure-Aware chunking** | M1 | `chunk_structure_aware()` | Dùng regex bóc tách các tiêu đề Markdown (`#`, `##`, `###`), gắn tên `section` vào metadata. Cực kỳ hiệu quả đối với tài liệu chính sách, sổ tay nhân sự, giúp giữ trọn vẹn các bảng biểu và danh sách kiểm tra trong đúng đề mục của nó. |
| **Vietnamese Word Tokenization** | M2 | `segment_vietnamese()` | Sử dụng `underthesea.word_tokenize(text, format="text")` rồi thay thế ký tự `_` bằng khoảng trắng. Điều này giải quyết triệt để lỗi lệch token giữa underthesea (tạo từ ghép `nghỉ_phép`) và BM25Okapi (tách theo khoảng trắng đơn thuần). |
| **BM25 + Dense Fusion (Hybrid Search)** | M2 | `reciprocal_rank_fusion()` | Thuật toán RRF với công thức $RRF(d) = \sum \frac{1}{k + rank + 1}$ ($k=60$) cho phép dung hòa hai thang đo điểm số hoàn toàn khác nhau (điểm tần suất BM25 và cosine similarity dense vector BAAI/bge-m3), giúp bắt trúng cả từ khóa mã hiệu văn bản lẫn ngữ nghĩa trừu tượng. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` đọc đồng thời cặp (query, document), tính tương tác chéo giữa từng token. Rút gọn từ top 20 candidate xuống top 3 context tinh hoa nhất, giảm thiểu nhiễu cho LLM và tăng Context Precision lên 0.8800. |
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Tự động hóa đánh giá hệ thống qua 4 chiều: Faithfulness (0.9000), Answer Relevancy (0.8500), Context Precision (0.8800), Context Recall (0.8845). Cây chẩn đoán lỗi phân loại chính xác nguyên nhân (do LLM bịa đặt hay do Retriever bỏ sót) và đề xuất hướng xử lý tương ứng. |
| **Contextual Prepend & Enrichment** | M5 | `contextual_prepend()`, `_enrich_single_call()` | Áp dụng kỹ thuật của Anthropic: bổ sung 1 câu tóm tắt vị trí và chủ đề vào đầu mỗi chunk trước khi embed. Tối ưu chi phí bằng phương thức `_enrich_single_call` gom cả 4 tác vụ (Summary, HyQA, Context, Metadata) vào 1 lần gọi API duy nhất. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình thực hành và tích hợp pipeline, tôi đã đối mặt và giải quyết các bài toán kỹ thuật thực tế sau:

### 1. Lỗi tokenization tiếng Việt trong BM25
- **Hiện tượng & Lỗi:** Khi tìm kiếm cụm từ `"nghỉ phép"`, BM25 ban đầu không trả về tài liệu chứa từ `"nghỉ phép"`.
- **Nguyên nhân:** Thư viện `underthesea` khi tách từ ghép tiếng Việt sẽ trả về chuỗi nối bằng dấu gạch dưới `nghỉ_phép`. Trong khi đó, `BM25Okapi` tokenize văn bản bằng cách `split(" ")`. Kết quả là văn bản có 1 token `"nghỉ_phép"`, còn câu truy vấn của người dùng nhập `"nghỉ phép"` được tách thành 2 token riêng lẻ `["nghỉ", "phép"]`. Sự bất đồng nhất token này khiến BM25 tính điểm số bằng 0.
- **Cách khắc phục:** Trong hàm `segment_vietnamese()`, thực hiện chuẩn hóa `.replace("_", " ")` ngay sau khi gọi `word_tokenize()`. Nhờ đó câu truy vấn và văn bản cùng chia sẻ không gian từ vựng đồng nhất.

### 2. Vấn đề xung đột thư viện Tokenizer của FlagEmbedding với Transformers phiên bản mới
- **Hiện tượng & Lỗi:** Sử dụng `FlagReranker` từ thư viện `FlagEmbedding` gây xung đột nghiêm trọng với các phiên bản `transformers >= 4.40` hoặc `5.0` (lỗi không tương thích với `XLMRobertaTokenizer`).
- **Cách giải quyết:** Chuyển sang sử dụng trực tiếp lớp `CrossEncoder` từ `sentence_transformers` (`from sentence_transformers import CrossEncoder`) với model `BAAI/bge-reranker-v2-m3`. Thư viện `sentence_transformers` quản lý vòng đời và tokenizer ổn định hơn, tương thích hoàn toàn với PyTorch và Transformers hiện hành.

### 3. Vấn đề bộ nhớ và không phụ thuộc Docker daemon cho Qdrant
- **Hiện tượng & Lỗi:** Môi trường local có thể chưa bật Docker daemon, dẫn đến `QdrantClient(host="localhost", port=6333)` bị lỗi kết nối socket.
- **Cách giải quyết:** Thiết lập cơ chế tự động fallback trong hàm `__init__` của `DenseSearch`: nếu không kết nối được tới server Qdrant qua cổng 6333, tự động chuyển sang chế độ in-memory `QdrantClient(":memory:")`. Điều này giúp toàn bộ test suite và pipeline chạy mượt mà độc lập trên mọi môi trường CI/CD hoặc máy cá nhân.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy định Nội bộ Doanh nghiệp (Enterprise Legal & Policy Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản: Chia đoạn cố định 500 ký tự (character-based text split), chỉ tìm kiếm bằng Vector Dense Search đơn thuần (OpenAI `text-embedding-3-small`), không có tầng Rerank.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Retrieval thất bại với thuật ngữ số hiệu:* Câu hỏi tra cứu số điều luật, số quyết định (VD: "Nghị định 13/2023/NĐ-CP", "Điều 15 khoản 2") vector search thường trượt hoặc lấy nhầm điều khoản khác.
  - *Xung đột phiên bản:* Khi quy chế công ty cập nhật phiên bản 2024, hệ thống vẫn trích xuất tài liệu 2023 đã hết hiệu lực.
  - *Hiện tượng đứt đoạn bảng biểu:* Các bảng phân cấp thẩm quyền tài chính bị cắt vụn khiến LLM đọc không hiểu được mối liên hệ giữa dòng và cột.

#### 2. Kế hoạch cải tiến với kiến trúc Production RAG

1. **Chunking Strategy:**
   - Kết hợp **Structure-Aware Chunking** (bóc tách theo cấu trúc Điều/Khoản/Mục của văn bản quy phạm) và **Hierarchical Chunking** (Parent 2048 ký tự chứa trọn vẹn một Điều luật, Child 256 ký tự trích từng Khoản chi tiết).
   - Đảm bảo khi người dùng tìm kiếm chi tiết một Khoản, hệ thống trả về toàn bộ ngữ cảnh của Điều luật đó cho LLM.

2. **Search Retrieval (Hybrid + RRF):**
   - Triển khai **Hybrid Search**: Kết hợp BM25 (với bộ từ điển pháp lý tiếng Việt và tách từ chuẩn qua `underthesea`) cùng Dense Vector `BAAI/bge-m3` (hỗ trợ đa ngữ và văn bản dài).
   - Hợp nhất kết quả bằng **RRF** ($k=60$) để đảm bảo các truy vấn chứa từ khóa đặc thù (mã số văn bản, mức phạt cụ thể) luôn được bắt trúng.

3. **Cross-Encoder Reranking:**
   - Tích hợp tầng rerank sử dụng `BAAI/bge-reranker-v2-m3` lấy top 20 candidate từ bước Hybrid Search và chọn lọc ra top 3 đoạn trích chính xác nhất.
   - Benchmark độ trễ bước Rerank duy trì dưới 120ms trên GPU/CPU tối ưu.

4. **Document Versioning & Metadata Filtering:**
   - Bổ sung trường metadata `document_version`, `effective_date`, `status` (active / superseded).
   - Tự động lọc các tài liệu cũ trừ khi người dùng có chủ đích hỏi so sánh lịch sử quy định.

5. **Continuous Evaluation với RAGAS:**
   - Xây dựng bộ test-set gồm 50 câu hỏi nghiệp vụ thực tế gắn kèm ground truth.
   - Thiết lập quy trình CI đánh giá tự động: Mỗi khi cập nhật kho tri thức hoặc thay đổi prompt, pipeline phải đạt tối thiểu `Faithfulness >= 0.85` và `Context Precision >= 0.80`.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1:** Tái cấu trúc cơ sở dữ liệu tri thức; áp dụng Structure-Aware và Hierarchical Chunking; gắn metadata phiên bản hiệu lực.
- **Tuần 2:** Triển khai Hybrid Search (BM25 tiếng Việt + Qdrant BGE-M3 + RRF); đánh giá cải thiện tỷ lệ bắt trúng từ khóa số hiệu.
- **Tuần 3:** Tích hợp Cross-Encoder Reranker; tối ưu hóa độ trễ; đóng gói API serving với FastAPI.
- **Tuần 4:** Chạy RAGAS benchmark toàn diện; xây dựng Diagnostic Tree giám sát production và triển khai hệ thống cho người dùng thử nghiệm nội bộ.
