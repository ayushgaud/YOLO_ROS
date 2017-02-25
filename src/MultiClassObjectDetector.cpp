//
//  MultiClassObjectDetector.cpp
//  pr2_perception
//
//  Created by Xun Wang on 12/05/16.
//  Copyright (c) 2016 Xun Wang. All rights reserved.
//

#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <darknet/image.h>
#include "MultiClassObjectDetector.h"

#include "dn_object_detect/DetectedObjects.h"

#include "std_msgs/String.h"

#include <sstream>
namespace uts_perp {

using namespace std;
using namespace cv;
  
static const int kPublishFreq = 10;
static const string kDefaultDevice = "/buglabugla/left/image_raw";
static const string kYOLOModel = "data/yolo.weights";
static const string kYOLOConfig = "data/yolo.cfg";

static const char * VoClassNames[] = { "aeroplane", "bicycle", "bird", // should not hard code these name
                              "boat", "bottle", "bus", "car",
                              "cat", "chair", "cow", "diningtable",
                              "dog", "horse", "motorbike",
                              "person", "pottedplant", "sheep",
                              "sofa", "train", "tvmonitor"
                            };

static int NofVoClasses = sizeof( VoClassNames ) / sizeof( VoClassNames[0] );

/*
extern "C" {
void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);

}
*/

MultiClassObjectDetector::MultiClassObjectDetector() :
  imgTrans_( priImgNode_ ),
  initialised_( false ),
  doDetection_( true ),
  debugRequests_( 0 ),
  srvRequests_( 0 ),
  procThread_( NULL )
{
}

MultiClassObjectDetector::~MultiClassObjectDetector()
{
}

void MultiClassObjectDetector::init()
{
  NodeHandle priNh( "~" );
  std::string yoloModelFile;
  std::string yoloConfigFile;
  
  priNh.param<std::string>( "camera", cameraDevice_, kDefaultDevice );
  priNh.param<std::string>( "yolo_model", yoloModelFile, kYOLOModel );
  priNh.param<std::string>( "yolo_config", yoloConfigFile, kYOLOConfig );
  priNh.param( "threshold", threshold_, 0.04f );
  
  const boost::filesystem::path modelFilePath = yoloModelFile;
  const boost::filesystem::path configFilepath = yoloConfigFile;
  
  if (boost::filesystem::exists( modelFilePath ) && boost::filesystem::exists( configFilepath )) {
    darkNet_ = parse_network_cfg( (char*)yoloConfigFile.c_str() );
    load_weights( darkNet_, (char*)yoloModelFile.c_str() );
    detectLayer_ = darkNet_->layers[darkNet_->n-1];
    printf( "detect layer side = %d n = %d\n", detectLayer_.side, detectLayer_.n );
    maxNofBoxes_ = detectLayer_.side * detectLayer_.side * detectLayer_.n;
    set_batch_network( darkNet_, 1 );
    srand(2222222);
  }
  else {
    ROS_ERROR( "Unable to find YOLO darknet configuration or model files." );
    return;
  }

  ROS_INFO( "Loaded detection model data." );
  initialised_ = true;

  imgSub_ = imgTrans_.subscribe( cameraDevice_, 1, &MultiClassObjectDetector::processingRawImages, this );

  dtcPub_ = priImgNode_.advertise<dn_object_detect::DetectedObjects>( "/dn_object_detect/detected_objects", 1 );
  //bb_pub = priImgNode_.advertise<std_msgs::String>("/bb_coordinates", 1);
  imgPub_ = imgTrans_.advertise("/yolo/image",1);

}

void MultiClassObjectDetector::fini()
{

  free_network( darkNet_ );
  initialised_ = false;
}

void MultiClassObjectDetector::continueProcessing()
{
  ros::spin();
}
  
void MultiClassObjectDetector::doObjectDetection()
{
  ros::Rate publish_rate( kPublishFreq );
  ros::Time ts;

  float nms = 0.5;

  box * boxes = (box *)calloc( maxNofBoxes_, sizeof( box ) );
  float **probs = (float **)calloc( maxNofBoxes_, sizeof(float *));
  for(int j = 0; j < maxNofBoxes_; ++j) {
    probs[j] = (float *)calloc( detectLayer_.classes, sizeof(float *) );
  }

  DetectedList detectObjs;
  detectObjs.reserve( 30 ); // silly hardcode

 
    
      if (imgMsgPtr_.get() == NULL) {
        publish_rate.sleep();
      }
      try {
        cv_ptr_ = cv_bridge::toCvCopy( imgMsgPtr_, sensor_msgs::image_encodings::BGR8 );
        ts = imgMsgPtr_->header.stamp;
      }
      catch (cv_bridge::Exception & e) {
        ROS_ERROR( "Unable to convert image message to mat." );
        imgMsgPtr_.reset();
        publish_rate.sleep();
      }
      imgMsgPtr_.reset();
    

    if (cv_ptr_.get()) {
      IplImage img = cv_ptr_->image;
      image im = ipl_to_image( &img );
      image sized = resize_image( im, darkNet_->w, darkNet_->h );
      float *X = sized.data;
      float *predictions = network_predict( darkNet_, X );

      //ROS_INFO( "Detction done" );

      //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
      convert_yolo_detections( predictions, detectLayer_.classes, detectLayer_.n, detectLayer_.sqrt, detectLayer_.side, 1, 1, threshold_, probs, boxes, 0);
      if (nms) {
        do_nms_sort( boxes, probs, maxNofBoxes_, detectLayer_.classes, nms );
      }
      //   std::cerr<<"no of classes  "<<detectLayer_.classes<<"\n";
      // for( int inc = 0; inc < detectLayer_.classes; inc++)
      //   std::cerr<<*(predictions+inc)<<"   ";

      this->consolidateDetectedObjects( &im, boxes, probs, detectObjs );


  cv::Scalar boundColour( 255, 0, 255 );
  cv::Scalar connColour( 209, 47, 27 );

  for (size_t i = 0; i < detectObjs.size(); i++) { 
    dn_object_detect::ObjectInfo obj = detectObjs[i];
	   // cv::rectangle( cv_ptr_->image, cv::Rect(obj.tl_x, obj.tl_y, obj.width, obj.height),
	   //     boundColour, 2 );

    // only write text on the head or body if no head is detected.
   // std::string box_text = format( "%s prob=%.2f", obj.type.c_str(), obj.prob );
    // Calculate the position for annotated text (make sure we don't
    // put illegal values in there):
   // cv::Point2i txpos( std::max(obj.tl_x - 10, 0),
    //                  std::max(obj.tl_y - 10, 0) );
    // And now put it into the image:
   // putText( cv_ptr_->image, box_text, txpos, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
  //cv_ptr_->image = cv_ptr_->image(cv::Rect(obj.tl_x, obj.tl_y, obj.width, obj.height));
    // std_msgs::String msg;
    // std::stringstream ss;
    // ss << obj.tl_x << " " << obj.tl_y << " " << obj.width << " " << obj.height;
    // msg.data = ss.str();
    // bb_pub.publish(msg);
    cv::Mat target(cv_ptr_->image.size(), cv_ptr_->image.type());
    target = cv::Scalar(255,255,255);
    cv::Mat subImage = target(cv::Rect(obj.tl_x, obj.tl_y, obj.width, obj.height));
	cv_ptr_->image(cv::Rect(obj.tl_x, obj.tl_y, obj.width, obj.height)).copyTo(subImage);
    sensor_msgs::ImagePtr msg_1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", target).toImageMsg();
    imgPub_.publish(msg_1);
  }

//imshow("result",cv_ptr_->image);
// waitKey(30);
      //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
      free_image(im);

      free_image(sized);


//  publish_rate.sleep();
  }
    cv_ptr_.reset();

  // clean up
  for(int j = 0; j < maxNofBoxes_; ++j) {
    free( probs[j] );
  }
  free( probs );
  free( boxes );
}
  
void MultiClassObjectDetector::processingRawImages( const sensor_msgs::ImageConstPtr& msg )
{
  ROS_INFO( "got IMage" );

  imgMsgPtr_ = msg;
  doObjectDetection();
}

void MultiClassObjectDetector::consolidateDetectedObjects( const image * im, box * boxes,
      float **probs, DetectedList & objList )
{
  //printf( "max_nofb %d, NofVoClasses %d\n", max_nofb, NofVoClasses );
  int objclass = 6;
  float prob = 0.0;
  int max_ele = 0;
  int max_index = 0;
  objList.clear();
for(int i = 0; i < maxNofBoxes_; ++i){
  prob = probs[i][objclass];
   if (boxes[i].w * boxes[i].h > max_ele && prob > threshold_){
    max_ele = boxes[i].w * boxes[i].h;
    max_index = i;}

}
 
    //objclass = max_index( probs[i], NofVoClasses );
    
    //if(objclass!=6)
    //	continue;
      prob = probs[max_index][objclass];
      if (prob > threshold_ && prob <= 1) {
      int width = pow( prob, 0.5 ) * 10 + 1;
      dn_object_detect::ObjectInfo newObj;
      newObj.type = VoClassNames[objclass];
      newObj.prob = prob;

      printf("%s: %.2f\n", VoClassNames[objclass], prob);
      /*
      int offset = class * 17 % classes;
      float red = get_color(0,offset,classes);
      float green = get_color(1,offset,classes);
      float blue = get_color(2,offset,classes);
      float rgb[3];
      rgb[0] = red;
      rgb[1] = green;
      rgb[2] = blue;
      box b = boxes[max_index];
      */

      int left  = (boxes[max_index].x - boxes[max_index].w/2.) * im->w;
      int right = (boxes[max_index].x + boxes[max_index].w/2.) * im->w;
      int top   = (boxes[max_index].y - boxes[max_index].h/2.) * im->h;
      int bot   = (boxes[max_index].y + boxes[max_index].h/2.) * im->h;

      if (right > im->w-1)  right = im->w-1;
      if (bot > im->h-1)    bot = im->h-1;

      newObj.tl_x = left < 0 ? 0 : left;
      newObj.tl_y = top < 0 ? 0 : top;
      newObj.width = right - newObj.tl_x;
      newObj.height = bot - newObj.tl_y;
      objList.push_back( newObj );
      //draw_box_width(im, left, top, right, bot, width, red, green, blue);
      //if (labels) draw_label(im, top + width, left, labels[class], rgb);
    }
  
}

} // namespace uts_perp
