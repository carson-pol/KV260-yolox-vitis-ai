#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <vitis/ai/yolovx.hpp>

static const char* COCO[] = {
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
  "toothbrush"
};

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <model_name> <input_pattern> <output_dir>\n"
              << "  e.g. " << argv[0]
              << " yolox_nano_ptq \"frames/f_%05d.jpg\" out_frames\n";
    return 1;
  }

  cv::VideoCapture cap(argv[2]);
  if (!cap.isOpened()) {
    std::cerr << "cannot open frame sequence: " << argv[2] << "\n";
    return 1;
  }

  auto yolo = vitis::ai::YOLOvX::create(argv[1], true);
  if (!yolo) { std::cerr << "cannot create model: " << argv[1] << "\n"; return 1; }

  cv::Mat frame;
  long n = 0;
  long total_dets = 0;
  double infer_total = 0.0;
  auto wall_start = std::chrono::steady_clock::now();

  while (cap.read(frame)) {
    auto t0 = std::chrono::steady_clock::now();
    auto results = yolo->run(frame);
    auto t1 = std::chrono::steady_clock::now();
    infer_total += std::chrono::duration<double>(t1 - t0).count();
    n++;
    total_dets += results.bboxes.size();

    for (auto& r : results.bboxes) {
      auto& b = r.box;
      cv::rectangle(frame, cv::Point(b[0], b[1]), cv::Point(b[2], b[3]),
                    cv::Scalar(0, 255, 0), 2);
      std::ostringstream lbl;
      lbl << (r.label >= 0 && r.label < 80 ? COCO[r.label] : "?")
          << " " << std::fixed << std::setprecision(2) << r.score;
      cv::putText(frame, lbl.str(), cv::Point(b[0], b[1] - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    std::ostringstream hud;
    hud << "YOLOX-Nano INT8 | DPUCZDX8G B4096 | "
        << std::fixed << std::setprecision(1) << (n / infer_total) << " FPS";
    cv::putText(frame, hud.str(), cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

    std::ostringstream p;
    p << argv[3] << "/o_" << std::setw(5) << std::setfill('0') << n << ".jpg";
    cv::imwrite(p.str(), frame);

    if (n % 30 == 0) std::cout << "frame " << n << "\r" << std::flush;
  }

  double wall = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_start).count();
  std::cout << "\nframes " << n
            << "  detections " << total_dets
            << "  inference-only FPS " << std::fixed << std::setprecision(2)
            << (n / infer_total)
            << "  end-to-end FPS " << (n / wall) << "\n";
  return 0;
}
