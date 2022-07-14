////////////////////////////////////////////////////////////////////////
// Include file for mp structures
////////////////////////////////////////////////////////////////////////
#ifndef __MP__
#define __MP__

#include <vector>

////////////////////////////////////////////////////////////////////////

struct MPImage {
  // Constructor/destructor
  MPImage(void);
  ~MPImage(void);

public:
  int house_index;
  struct MPPanorama *panorama;
  int panorama_index;
  char *name, *depth_filename, *color_filename;
  int camera_index;
  int yaw_index;
  double extrinsics[16];
  double intrinsics[9];
  int width, height;
  double position[3];
};



////////////////////////////////////////////////////////////////////////

struct MPPanorama {
  // Constructor/destructor
  MPPanorama(void);
  ~MPPanorama(void);

  // Manipulation stuff
  void InsertImage(MPImage *image);
  void RemoveImage(MPImage *image);

public:
  int house_index;
  struct MPRegion *region;
  int region_index;
  char *name;
  std::vector<MPImage *> images;
};


////////////////////////////////////////////////////////////////////////

struct MPRegion {
  // Constructor/destructor
  MPRegion(void);
  ~MPRegion(void);


  // Manipulation stuff
  void InsertPanorama(MPPanorama *panorama);
  void RemovePanorama(MPPanorama *panorama);

public:
  int house_index;
  int level_index;
  std::vector<MPPanorama*> panoramas;
  char *label;
};

class MP_Parser {

    public:

        MP_Parser(const char *filename);
        ~MP_Parser();

        std::vector<MPRegion*> regions;
        std::vector<MPPanorama*> panoramas;
        std::vector<MPImage*> images;

    private:
        void insertRegion(MPRegion* region);
        void insertPanorama(MPPanorama* panorama);
        void insertImage(MPImage* image);
        int parseHouseFile(const char *filename);
};

#endif