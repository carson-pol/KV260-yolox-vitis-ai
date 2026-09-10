#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vitis/ai/yolovx.hpp>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <model_name> <input.jpg> <output.jpg>\n";
    return 1;
  }
  const std::string model_name = argv[1];
  const std::string in_file = argv[2];
  const std::string out_file = argv[3];

  cv::Mat img = cv::imread(in_file);
  if (img.empty()) {
    std::cerr << "cannot read image: " << in_file << "\n";
    return 1;
  }

  auto yolo = vitis::ai::YOLOvX::create(model_name, true);
  if (!yolo) {
    std::cerr << "failed to create model: " << model_name << "\n";
    return 1;
  }

  auto results = yolo->run(img);
  std::cout << "detections: " << results.bboxes.size() << "\n";

  for (auto& r : results.bboxes) {
    auto& b = r.box;
    std::cout << "label " << r.label << "  score " << std::fixed
              << std::setprecision(4) << r.score << "  box "
              << std::setprecision(1) << b[0] << " " << b[1] << " " << b[2]
              << " " << b[3] << "\n";
    cv::rectangle(img, cv::Point(b[0], b[1]), cv::Point(b[2], b[3]),
                  cv::Scalar(0, 255, 0), 2);
  }

  cv::imwrite(out_file, img);
  std::cout << "wrote " << out_file << "\n";
  return 0;
}
