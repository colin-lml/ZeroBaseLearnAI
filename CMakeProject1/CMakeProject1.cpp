// CMakeProject1.cpp: 定义应用程序的入口点。
//

#include "CMakeProject1.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

#include "torch/torch.h"
#if 0
#include <io.h>
void load_data_from_folder(std::string path, std::string type, std::vector<std::string>& list_images, std::vector<int>& list_labels, int label);

void load_data_from_folder(std::string path, std::string type, std::vector<std::string>& list_images, std::vector<int>& list_labels, int label)
{
    /*
     * path：文件夹地址
     * type：图片类型
     * list_images：所有图片的名称
     * list_label：各个图片的标签，也就是所属的类
     * label：类别的个数
    */
    long long hFile = 0; //句柄
    struct _finddata_t fileInfo;// 记录读取到文件的信息
    std::string pathName;

    // 调用_findfirst函数,其第一个参数为遍历的文件夹路径，*代表任意文件。注意路径最后,需要添加通配符
    // 如果失败,返回-1,否则,就会返回文件句柄,并且将找到的第一个文件信息放在_finddata_t结构体变量中
    if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1)
    {
        return;
    }
    // 通过do{}while循环,遍历所有文件
    do
    {
        const char* filename = fileInfo.name;// 获得文件名
        const char* t = type.data();

        if (fileInfo.attrib & _A_SUBDIR) //是子文件夹
        {
            //遍历子文件夹中的文件(夹)
            if (strcmp(filename, ".") == 0 || strcmp(filename, "..") == 0) //子文件夹目录是.或者..
                continue;
            std::string sub_path = path + "\\" + fileInfo.name;// 增加多一级
            label++;
            load_data_from_folder(sub_path, type, list_images, list_labels, label);// 读取子文件夹的文件

        }
        else //判断是不是后缀为type文件
        {
            if (strstr(filename, t))
            {
                std::string image_path = path + "\\" + fileInfo.name;// 构造图像的地址
                list_images.push_back(image_path);
                list_labels.push_back(label);
            }
        }
        //其第一个参数就是_findfirst函数的返回值,第二个参数同样是文件信息结构体
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
}
class myDataset :public torch::data::Dataset<myDataset> 
{
public:
    int num_classes = 0;
    myDataset(std::string image_dir, std::string type) 
    {
        // 调用遍历文件的函数
        load_data_from_folder(image_dir, std::string(type), image_paths, labels, num_classes);
    }
    // 重写 get()，根据传入的index来获得指定的数据
    torch::data::Example<> get(size_t index) override 
    {
        std::string image_path = image_paths.at(index);// 根据index得到指定的图像
        cv::Mat image = cv::imread(image_path);// 读取图像
        cv::resize(image, image, cv::Size(224, 224));// 调整大小，使得尺寸统一，用于张量stack
        int label = labels.at(index);//
        // 将图像数据转换为张量image_tensor，尺寸{image.rows, image.cols, 3}，元素的数据类型为byte
        // Channels x Height x Width
        torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });
        //
        torch::Tensor label_tensor = torch::full({ 1 }, label);
        return { img_tensor.clone(), label_tensor.clone() };// 返回图像及其标签
    }
    // Return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    };
private:
    std::vector<std::string> image_paths;// 所有图像的地址
    std::vector<int> labels;// 所有图像的类别
};

#endif


struct Net : torch::nn::Module
{
    Net()
    {
       auto l = torch::nn::Linear(2, 2);
       auto l2 = torch::nn::Linear(2, 2);

       l->weight.data() = torch::tensor({ {0.15,0.25}
                                          ,{0.2,0.3} });
                                        

       l->bias.data() = torch::tensor({0.35,0.35});

       l2->weight.data() = torch::tensor({ {0.4,0.5}
                                           ,{0.45,0.55} });
                                       
       float xxx = 0.15 * 0.05 + 0.2 * 0.1 + 0.35;
       float xxx2 = 0.25 * 0.05 + 0.3 * 0.1 + 0.35;

       std::cout << "l: " << xxx <<" l2: "<< xxx2 << std::endl;


       l2->bias.data() = torch::tensor({ 0.6,0.6 });
       
       //std::cout << "l:" << l->weight << std::endl;
      // std::cout << "l2:" << l2->weight << std::endl;

       fc1 = register_module("fc1", l);
       fc2 = register_module("fc2", l2);

    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor t = fc1->forward(x);
        std::cout << "t:" << t << std::endl;
        x = torch::sigmoid(t);

        x = torch::sigmoid(fc2->forward(x));

        return x;
    }

    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

int main()
{
    /*
    auto batch_size = 1;
    auto mydataset = myDataset("D:\\hymenoptera_data\\", ".jpg").map(torch::data::transforms::Stack<>());
    auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(mydataset), batch_size);

    for (auto& batch : *mdataloader) 
    {
        auto data = batch.data;
        auto target = batch.target;
        std::cout << data.sizes() << target << std::endl;
    }
 */

    //torch::Tensor a = torch::tensor(4, 8, torch::dtype(torch::kFloat32));
    //torch::Tensor b = torch::tensor({ {5, 6}, {7, 8} }, torch::dtype(torch::kFloat32));
    //torch::Tensor c = a + b;
    //std::cout << "xxx:\n" << c << std::endl;
   
    torch::Tensor input = torch::tensor({ { 0.05, 0.1 } });
   // std::cout << "xxx:\n" << input << std::endl<< input.sizes()<< std::endl;
    Net m;
    torch::Tensor v = m.forward(input);
    std::cout << "r : " << v << std::endl;

    std::cin.get();

	return 0;
}


