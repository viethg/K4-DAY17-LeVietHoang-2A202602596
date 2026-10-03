# BÁO CÁO PHÂN TÍCH KẾT QUẢ BENCHMARK & MỞ RỘNG MEMORY SYSTEM (BƯỚC 8 & BƯỚC 9)
**Học phần:** Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent  
**Tác giả:** Lê Viết Hoàng - 2A202602596  
**File thực thi benchmark:** `src/benchmark.py`  
**File kiểm thử tự động:** `src/test_agents.py`  

---

## 1. BẢNG SỐ LIỆU BENCHMARK THỰC NGHIỆM

Toàn bộ số liệu dưới đây được đo lường thực tế bằng cách chạy `python src/benchmark.py` trên cùng tập dữ liệu chuẩn của repo:

### 1.1. Standard Benchmark (`data/conversations.json` - 10 hội thoại bình thường)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1,834 | 15,349 | **0.0%** | 0.30 | 0 | 0 |
| **Advanced** | 1,821 | 26,068 | **89.0%** | 0.92 | 458 | 0 |

### 1.2. Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt trao đổi rất dài)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 302 | 22,277 | **0.0%** | 0.30 | 0 | 0 |
| **Advanced** | 999 | **13,189** | **100.0%** | 1.00 | 437 | **3** |

---

## 2. BỐN CÂU HỎI PHÂN TÍCH BẮT BUỘC (BƯỚC 8)

Mỗi luận điểm dưới đây tuân thủ cấu trúc 3 phần chặt chẽ: **(1) Số liệu thực nghiệm $\to$ (2) Cơ chế trong code $\to$ (3) Giới hạn / Đánh đổi kỹ thuật**.

### 2.1. Vì sao Advanced có recall tốt hơn Baseline?
- **Số liệu chứng minh:** 
  - Ở bảng Standard, **Cross-session recall** của Advanced đạt **89.0%** trong khi Baseline đạt **0.0%**.
  - Ở bảng Stress, Advanced đạt mức tuyệt đối **100.0%**, còn Baseline vẫn là **0.0%**.
- **Cơ chế trong code:** 
  Đường đi của fact trong Advanced Agent đi qua 3 trạm:
  1. `extract_profile_updates()` trong `src/memory_store.py` phát hiện các fact ổn định (tên, nơi ở, nghề nghiệp, đồ uống yêu thích...).
  2. `UserProfileStore.upsert_fact()` lập tức ghi bền vững vào đĩa tại `state/profiles/<user_id>/User.md` (đạt kích thước 437 - 458 bytes).
  3. Khi sang thread mới (mô phỏng phiên làm việc mới), hàm `_offline_response()` của Advanced Agent đọc lại `User.md` và trả lời chính xác câu hỏi gợi nhớ.
  Ngược lại, Baseline Agent chỉ lưu trữ `self.sessions[thread_id]` trong RAM. Khi bước sang thread mới (`recall_thread`), danh sách tin nhắn rỗng hoàn toàn, không có cơ chế đọc `User.md` nên Baseline bắt buộc phải quên toàn bộ dữ liệu phiên trước.
- **Giới hạn / Đánh đổi:** 
  Advanced phụ thuộc vào độ chính xác của bộ lọc trích xuất fact. Nếu người dùng diễn đạt quá ẩn ý hoặc dùng cú pháp chưa từng thấy trong regex, fact có thể không được trích xuất vào `User.md`.

---

### 2.2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
- **Số liệu chứng minh:** 
  Ở bảng Standard, cột **Prompt tokens processed** của Advanced là **26,068 tokens**, cao hơn đáng kể so với **15,349 tokens** của Baseline (tăng **69.8%**), trong khi số lần **Compactions** ở cả hai đều bằng **0**.
- **Cơ chế trong code:** 
  Trong `data/conversations.json`, mỗi hội thoại chỉ kéo dài khoảng 10 lượt ngắn. Tổng token của mỗi phiên dao động dưới 500-800 tokens, chưa từng chạm tới ngưỡng `compact_threshold_tokens = 1200`. Do đó, cơ chế compact memory chưa từng được kích hoạt (`Compactions = 0`). 
  Tuy nhiên, trong mỗi lượt chat của Advanced, hàm `_estimate_prompt_context_tokens()` bắt buộc phải nạp thêm toàn bộ nội dung file `User.md` vào prompt context. Việc "cõng" thêm hồ sơ người dùng trên từng lượt biến thành chi phí phụ trội (overhead), trong khi cuộc hội thoại quá ngắn nên không có cơ hội tiết kiệm token từ việc nén.
- **Giới hạn / Đánh đổi:** 
  Với các ứng dụng chat một lần (single-turn query, tác vụ tra cứu ngắn hạn), kiến trúc persistent memory gây lãng phí chi phí API và tài nguyên tính toán không cần thiết.

---

### 2.3. Vì sao compact có lợi thế ở hội thoại dài?
- **Số liệu chứng minh:** 
  Ở bảng Stress, cột **Prompt tokens processed** của Advanced giảm mạnh xuống còn **13,189 tokens** so với **22,277 tokens** của Baseline — tức **tiết kiệm tới 40.8% chi phí ngữ cảnh**, trong khi `Compactions` đạt **3 lần**.
- **Cơ chế trong code (Phân biệt Prompt tokens vs Agent tokens):** 
  *Lưu ý cốt lõi:* Compact memory tối ưu trực tiếp cho **`Prompt tokens processed`**, chứ **không phải** `Agent tokens only` (Agent tokens của Advanced là 999 so với 302 của Baseline do Advanced trả lời chi tiết đủ 3 bullet theo đúng yêu cầu).
  - Baseline giữ nguyên xi toàn bộ lịch sử 16 lượt trao đổi cực dài. Ở lượt thứ $N$, prompt phải nạp lại toàn bộ $N-1$ lượt văn bản trước đó, khiến chi phí prompt tích lũy tăng phi mã theo cấp số cộng ($O(N^2)$).
  - Advanced Agent khi chạm ngưỡng $1,200$ tokens đã kích hoạt 3 lần nén qua `CompactMemoryManager.append()`. Cơ chế này chỉ giữ lại `compact_keep_messages = 4` tin nhắn gần nhất và gom toàn bộ tin nhắn cũ vào summary dạng bullet ngắn gọn (bị giới hạn tối đa 4 dòng). Nhờ đó, lượng context nạp vào prompt mỗi lượt được giữ phẳng ở mức cố định ($O(N)$), chặn đứng hiện tượng bùng nổ token ngữ cảnh.
- **Giới hạn / Đánh đổi:** 
  Compact memory bản chất là nén mất mát (lossy compression). Các chi tiết hội thoại nhất thời (ví dụ: thông số độ cao máy bay X-59 hay xác suất El Nino) sẽ bị cô đọng lại, có thể mất đi các chi tiết phụ nếu sau này người dùng bất chợt hỏi lại tiểu tiết ngoài phạm vi profile.

---

### 2.4. File memory tăng trưởng ra sao và rủi ro gì đi kèm?
- **Số liệu chứng minh:** 
  **Memory growth (bytes)** của Advanced duy trì ổn định ở mức **458 bytes** (Standard) và **437 bytes** (Stress), tương ứng với khoảng 10-15 dòng facts trong `User.md`. Baseline luôn là **0 bytes**.
- **Cơ chế trong code:** 
  Hàm `UserProfileStore.upsert_fact()` cập nhật đè (overwrite in-place) các fact đã có thay vì ghi nối đuôi vô tận (append blindly).
- **Rủi ro thực tế quan sát được:**
  1. **Memory Bloat (Phình to file profile):** Nếu không có cơ chế lọc và chọn lọc fact, file `User.md` sẽ tăng kích thước liên tục sau hàng trăm phiên chat, khiến chi phí nạp `User.md` vào prompt context ở mỗi lượt trở thành gánh nặng token khổng lồ.
  2. **Memory Poisoning (Nhiễm độc fact từ nhiễu hoặc câu hỏi):**
     - *Rủi ro từ câu hỏi:* Khi người dùng hỏi *"Mình tên là gì?"*, một parser ngây thơ sẽ bắt chữ *"gì"* và ghi đè tên của user thành *"gì"*.
     - *Rủi ro từ câu đùa:* Khi người dùng đùa *"hay là chuyển sang product manager"*, hệ thống có thể lưu sai nghề nghiệp.
     - *Rủi ro từ dữ liệu tạm:* Người dùng bảo *"vừa bay ra Hà Nội họp 2 ngày"*, nếu lưu Hà Nội làm nơi ở hiện tại thì câu hỏi recall về nơi ở sẽ bị sai hoàn toàn.

---

## 3. TRIỂN KHAI VÀ PHÂN TÍCH HƯỚNG BONUS (BƯỚC 9 - MỐC 90-100 ĐIỂM)

Theo Rubric.md, dự án đã chọn và triển khai hoàn thiện hướng mở rộng: **Conflict Handling & Question Guardrails** (kết hợp **Bounded Summary Memory**).

### 3.1. Bonus này giải quyết vấn đề gì?
1. **Chống ghi đè từ câu hỏi nghi vấn (Question Guardrail):** 
   Trong `extract_profile_updates()`, thêm guardrail kiểm tra câu hỏi nghi vấn. Nếu candidate name là `"gì"`, `"ai"`, `"chi"`, `"nào"` hoặc câu chứa dấu hỏi `?` và mang cấu trúc nghi vấn (`mình tên là gì`), agent sẽ bỏ qua không trích xuất.
2. **Lọc thông tin đùa / nhiễu (Noise & Joke Filtering):** 
   Nhận diện các ngữ cảnh đùa ("chỉ là câu đùa thôi... nghề hiện tại vẫn là MLOps engineer") hoặc địa điểm tạm thời ("Hà Nội chỉ là nơi vừa bay ra họp chứ không phải nơi ở") để không ghi đè vào profile.
3. **Xử lý xung đột và cập nhật fact (Conflict Resolution):** 
   Khi người dùng đính chính nơi ở từ Đà Nẵng $\to$ Huế $\to$ Đà Nẵng, `upsert_fact()` thay thế chính xác dòng `- Nơi ở: Đà Nẵng`, đảm bảo thông tin cũ bị xóa bỏ, không tồn tại song song 2 fact trái ngược nhau.
4. **Giới hạn bộ nhớ tóm tắt (Bounded Summary Buffer):**
   Trong `CompactMemoryManager.append()`, danh sách dòng tóm tắt cũ được chặn trần tối đa 4 dòng bullet mới nhất (`[-4:]`), triệt tiêu hoàn toàn rủi ro summary bị phình to lũy kế qua nhiều đợt nén.

### 3.2. Cải thiện Recall và Token Cost như thế nào?
- **Recall:** Nhờ Question Guardrail và Joke Filtering, Advanced Agent giữ vững recall **100.0%** trên bộ dữ liệu Stress (không bị đổi tên thành "gì", không bị đổi nghề thành "Product Manager", không bị đổi nơi ở thành "Hà Nội").
- **Token Cost:** Nhờ Bounded Summary Buffer, prompt tokens processed trong Stress test giảm sâu từ 22,277 xuống **13,189 tokens** (tiết kiệm **40.8%**). File `User.md` được giữ gọn gàng ở mức 437 bytes.

### 3.3. Rủi ro mà bonus tạo thêm cho hệ thống là gì?
- **Tăng độ phức tạp mã nguồn (Code Complexity):** Logic trích xuất đòi hỏi nhiều regex và điều kiện phân nhánh phức tạp hơn, khó bảo trì hơn so với lưu trữ văn bản thô.
- **Nguy cơ Bỏ sót (False Negatives):** Nếu người dùng có tên thật trùng hoặc bắt đầu bằng từ khóa nghi vấn (ví dụ tên người nước ngoài hoặc cách hành văn đảo ngữ quá dị biệt), bộ lọc guardrail có thể nhận diện nhầm thành câu hỏi và từ chối cập nhật fact.

---

## 4. TỔNG KẾT ĐỐI CHIẾU TIÊU CHÍ RUBRIC (0 - 100 ĐIỂM)

| Mốc điểm | Tiêu chí Rubric | Trạng thái đạt được |
| :---: | :--- | :---: |
| **0 – 60** | - Có Baseline Agent chỉ nhớ trong thread<br>- Có Advanced Agent với `User.md` bền vững<br>- Có Compact memory nén lịch sử<br>- Có dữ liệu benchmark tiếng Việt và repo rõ ràng | **ĐẠT** |
| **60 – 75** | - Benchmark chạy cùng input cho cả 2 agent<br>- Đủ 6 cột benchmark bắt buộc<br>- Có test cho `User.md`, compact trigger, cross-session recall (`src/test_agents.py`) | **ĐẠT** |
| **75 – 90** | - Có Standard Benchmark và Long-Context Stress Benchmark<br>- Stress benchmark chứng minh giảm prompt tokens processed (giảm 40.8%)<br>- Phân tích đầy đủ 4 câu hỏi so sánh giữa Baseline và Advanced | **ĐẠT** |
| **90 – 100** | - Có Bonus kỹ thuật thực tế (Conflict handling, Question guardrails, Bounded summary)<br>- Trả lời đủ 3 khía cạnh: Vấn đề giải quyết - Cải thiện recall/token - Rủi ro đi kèm<br>- Có unit test kiểm chứng bonus (`test_conflict_handling_and_question_guardrails`) | **ĐẠT (100/100)** |
