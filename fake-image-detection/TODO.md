# Kế hoạch triển khai Phân loại Ảnh Thật / Giả (Real vs Fake)

Dự án này ban đầu dự định sử dụng phương pháp DIRE, nhưng do hạn chế về thời gian tính toán mô phỏng (Inversion & Reconstruction), chúng ta đã chuyển sang phương pháp hiệu quả hơn là **Fine-tune mô hình Deep Learning (ResNet50 / ViT)** trực tiếp trên tập ảnh gốc.

Dưới đây là tài liệu kỹ thuật chi tiết giải thích cho từng phương pháp tiếp cận và danh sách các nhiệm vụ kế hoạch.

---

## 📚 TÀI LIỆU KỸ THUẬT (DOCUMENTATION)

### 1. Phương pháp DIRE (Diffusion Reconstruction Error)

**DIRE** là một phương pháp phát hiện ảnh do AI tạo ra dựa trên giả thuyết tự nhiên: *Bất kỳ bức ảnh nào được tạo ra bởi một mô hình khuếch tán (Diffusion Model) sẽ dễ dàng được chính mô hình đó tái tạo lại với sai số cực thấp so với ảnh thật (ảnh tự nhiên).*

**Công thức toán học & Các bước thực hiện:**
Giả sử chúng ta có một bức ảnh đầu vào là $x_0$.

1. **Quá trình Đảo ngược (DDIM Inversion):**
   Thay vì chạy từ nhiễu tiềm ẩn ra ảnh, ta đi lùi quá trình sinh ảnh của DDIM để tìm lại biểu diễn nhiễu (latent noise) $x_T$ ở bước thời gian lớn nhất $T$.
   $$x_{t+1} = \sqrt{\alpha_{t+1}} \underbrace{\left( \frac{x_t - \sqrt{1 - \alpha_t}\epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right)}_{\text{Dự đoán } x_0} + \sqrt{1 - \alpha_{t+1}}\epsilon_\theta(x_t, t)$$
   
2. **Quá trình Tái tạo (Reconstruction):**
   Ngay từ biểu diễn nhiễu $x_T$ vừa tìm được, ta dùng bước khử nhiễu (denoising) tiến về lại để sinh ra một bức ảnh mới, gọi là ảnh tái tạo $x'_0$.
   $$x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1 - \alpha_t}\epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right) + \sqrt{1 - \alpha_{t-1}}\epsilon_\theta(x_t, t)$$

3. **Tính toán Lỗi Tái tạo (Reconstruction Error):**
   Sai số tái tạo (DIRE error map) được tính ở cấp độ từng điểm ảnh (pixel) bằng giá trị tuyệt đối của hiệu số giữa ảnh được đưa vào và ảnh được cỗ máy tái tạo:
   $$E = |x_0 - x'_0|$$
   
4. **Đánh giá & Phân loại (Classification):**
   - **Ảnh AI (Fake):** Giá trị tổng $E$ sẽ rất nhỏ (Bản đồ lỗi thường tối nhạt). Lý do là vì ảnh ban đầu vốn đã nằm sẵn trong không gian phân phối học được của mô hình, nên nó đi luồn lách qua các lớp rất dễ dàng và mượt mà.
   - **Ảnh thật (Real):** Giá trị tổng $E$ sẽ cực lớn (Bản đồ lỗi rất sáng). Máy chưa từng thấy các cấu trúc hạt nhiễu tự nhiên này, nên khi nó cố tái tạo lại, nó sinh ra một mớ lộn xộn.
   - Khi có được không gian lỗi $E$, ta chỉ việc truyền nó qua mạng CNN siêu nhẹ (như ResNet-20) để cho ra kết quả cuối cùng.

👉 *Lưu ý:* Độ chính xác của DIRE rất ấn tượng vì nó đi vào tận gốc rễ của ảnh fake tạo bởi Diffusion model. Tuy nhiên, phương pháp này **quá chậm** trên các máy tính cá nhân (phải chạy lùi hàng chục step rồi tiến hàng chục step qua cỗ máy Txt2Img khổng lồ cho MỖI bức ảnh). Do đó, chúng ta thực thi **ResNet/ViT** (Phương pháp 2).

---

### 2. Phương pháp Tinh chỉnh Trực tiếp (ResNet / ViT)

Đây là phương pháp **Phân loại nhị phân (Binary Classification)** truyền thống chạy trực tiếp trên các kênh màu RGB. AI sẽ nhìn vào cấu trúc siêu nhỏ của pixel để tìm sự khác biệt (ví dụ như những vết răng cưa hoặc nhiễu không tự nhiên mờ mờ ở viền ảnh).

**1. Kiến trúc ResNet50 (Residual Networks):**
- **Đặc điểm:** Sử dụng các "Khối kết nối tắt" (Residual blocks / Skip Connections) được định nghĩa là $\mathcal{H}(x) = \mathcal{F}(x) + x$, giúp giải quyết vấn đề triệt tiêu đạo hàm (vanishing gradient) trong các mạng nơ-ron có độ sâu lớn. Nhờ tích chập (Convolution), ResNet rất nhạy cảm với các "nhiễu tần số cao" (high-frequency artifacts).
- **Cách tiến hành (Transfer Learning):**
  1. **Feature Extraction:** Tải trọng số chuẩn `IMAGENET1K_V1`. Mạng này đã được huấn luyện trên hàng triệu ảnh nên có sẵn các bộ lọc bén nhọn để cắt viền, tách khối tĩnh học.
  2. **Freeze Backbone:** Đóng băng toàn bộ các lớp học (Convolutional Layers) để giữ nguyên hiểu biết của nó, tiết kiệm VRAM tối đa khi huấn luyện.
  3. **Fine-Tuning:** Cắt bỏ lớp Classifier (Fully Connected) chứa 1000 đầu ra ban đầu. Thay bằng 2 chiều (Real và Fake). Thuật toán AdamW sẽ chỉ cập nhật các mũi tên kết nối liên quan tới lớp Classifier này.

**2. Kiến trúc ViT (Vision Transformer) - Hướng nâng cấp:**
- Khác với tính cục bộ vùng miền (local receptive field) của ResNet, ViT chặt bức hình $H \times W$ thành $N$ ma trận mảnh chắp vá (patches).
- Xuyên suốt mô hình là cơ chế Self-Attention (Sự chú ý tự thân) tính theo công thức: 
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- ViT cho **tầm nhìn toàn cục (global context)** kết nối các mảnh hoạ tiết cách rất xa nhau về mặt vật lý, từ đó nhận ra được những lề luật phối cảnh phi tuyến tính (ví dụ như cái bóng râm phản chiếu, hay luồng sáng bị sai của tay chân ở ảnh Fake). Tốc độ chậm nhưng mức độ thông minh cực kì đáng kể.

---

## ✅ Kế hoạch các bước thực hiện

*Toàn bộ quá trình triển khai Phương Pháp ResNet đã hoàn tất và nằm trong `main.ipynb`.*

- [x] **1. Chuẩn bị Dữ liệu & Môi trường:** Quét thư mục ảnh thật/giả. Phân chia Train/Val/Test bằng Stratified (70% - 15% - 15%).
- [x] **2. Data Pipeline & Augmentations (PyTorch DataLoader):** Resize ảnh chuẩn $224 \times 224$. Áp dụng lật ngang và dao động màu sắc nhẹ (`ColorJitter`) để đa dạng hoá dữ liệu luyện.
- [x] **3. Khởi tạo Mô hình (Transfer Learning):** Tải model, đẩy lên GPU (`cuda` hoặc `mps`). Kỹ thuật đóng băng backbone và thay đổi hàm mất mát.
- [x] **4. Vòng lặp Huấn luyện (Training):** Sử dụng hàm Loss `CrossEntropyLoss` cùng Optimizer `AdamW`. Bật tqdm để ghi nhận chạy thông số loss.
- [x] **5. Đánh giá (Evaluation):** Trích xuất Confusion Matrix, kiểm tra độ chênh lệch loss qua Validation/Test Loader. Tránh Overfitting.
- [x] **6. Lưu Mô hình & Ứng dụng (Inference):** Lưu trọng số ra đuôi file `resnet50_best_real_fake.pth`. Đóng gói hàm truyền 1 bức ảnh vào dự đoán.
