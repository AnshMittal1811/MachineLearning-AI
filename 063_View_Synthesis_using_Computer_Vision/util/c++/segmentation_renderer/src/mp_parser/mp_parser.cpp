////////////////////////////////////////////////////////////////////////
// Source file for mp structures
////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////
#include "mp_parser.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

////////////////////////////////////////////////////////////////////////
// IMAGE MEMBER FUNCTIONS
////////////////////////////////////////////////////////////////////////

MPImage::
MPImage(void)
  : house_index(-1),
    panorama(NULL),
    panorama_index(-1),
    name(NULL),
    camera_index(-1),
    yaw_index(-1),
    extrinsics{1,0,0,0,0,1,0,0, 0,0,1,0,0,0,0,1},
    intrinsics{1,0,0,0,1,0,0,0,1},
    width(0), height(0),
    position{0,0,0}
{
}



MPImage::
~MPImage(void)
{
  // Remove from panorama and house
  if (panorama) panorama->RemoveImage(this);

  // Delete names
  if (name) free(name);
}

////////////////////////////////////////////////////////////////////////
// PANORAMA MEMBER FUNCTIONS
////////////////////////////////////////////////////////////////////////

MPPanorama::
MPPanorama(void)
  : house_index(-1),
    region(NULL),
    region_index(-1),
    name(NULL),
    images()
{
}



MPPanorama::
~MPPanorama(void)
{
  // Remove all images
  while (!images.size() == 0) RemoveImage(images.back());
  
  // Remove from region and house
  if (region) region->RemovePanorama(this);
}



void MPPanorama::
InsertImage(MPImage *image)
{
  // Insert image
  image->panorama = this;
  image->panorama_index = images.size();
  images.push_back(image);
}



void MPPanorama::
RemoveImage(MPImage *image)
{
  // Remove image
  MPImage *tail = images.back();
  tail->panorama_index = image->panorama_index;
  images[image->panorama_index] = tail;
  images.pop_back();
  image->panorama = NULL;
  image->panorama_index = -1;
}

////////////////////////////////////////////////////////////////////////
// REGION MEMBER FUNCTIONS
////////////////////////////////////////////////////////////////////////

MPRegion::
MPRegion(void)
  : house_index(-1),
    level_index(-1),
    panoramas(),
    label(NULL)
{
}



MPRegion::
~MPRegion(void)
{
  // Remove panoramas, surfaces, and portals
  while (panoramas.size() > 0) RemovePanorama(panoramas.back());

  // Delete label
  if (label) free(label);
}

void MPRegion::
InsertPanorama(MPPanorama *panorama)
{
  // Insert panorama
  panorama->region = this;
  panorama->region_index = panoramas.size();
  panoramas.emplace_back(panorama);
}



void MPRegion::
RemovePanorama(MPPanorama *panorama)
{
  // Remove panorama
  MPPanorama *tail = panoramas.back();
  tail->region_index = panorama->region_index;
  panoramas[panorama->region_index] = tail;
  panoramas.pop_back();
  panorama->region = NULL;
  panorama->region_index = -1;
}

////////////////////////////////////////////////////////////////////////
// Ascii file parsing
////////////////////////////////////////////////////////////////////////

MP_Parser::MP_Parser(const char *filename) {
    parseHouseFile(filename);
}

MP_Parser::~MP_Parser() = default;

int MP_Parser::parseHouseFile(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open house file %s\n", filename);
    return 0;
  }

  // Useful variables
  char cmd[1024], version[1024], name_buffer[1024], label_buffer[1024];
  int nimages, npanoramas, nvertices, nsurfaces, nsegments, nobjects, ncategories, nregions, nportals, nlevels;
  int house_index, level_index, region_index, surface_index, category_index, object_index, panorama_index, id, dummy;
  double height, area;
  double position[3];
  double normal[3];
  double box[3][3];

  // Read file type and version
  fscanf(fp, "%s%s", cmd, version);
  if (strcmp(cmd, "ASCII")) {
    fprintf(stderr, "Unable to read ascii file %s, wrong type: %s\n", filename, cmd);
    return 0;
  }

  // Read header
  if (!strcmp(version, "1.0")) {
    nsegments = 0;
    nobjects = 0;
    ncategories = 0;
    nportals = 0;
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%s", name_buffer);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%d", &nimages);
    fscanf(fp, "%d", &npanoramas);
    fscanf(fp, "%d", &nvertices);
    fscanf(fp, "%d", &nsurfaces);
    fscanf(fp, "%d", &nregions);
    fscanf(fp, "%d", &nlevels);
    fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int i = 0; i < 8; i++) fscanf(fp, "%d", &dummy);
  }
  else {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%s", name_buffer);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%d", &nimages);
    fscanf(fp, "%d", &npanoramas);
    fscanf(fp, "%d", &nvertices);
    fscanf(fp, "%d", &nsurfaces);
    fscanf(fp, "%d", &nsegments);
    fscanf(fp, "%d", &nobjects);
    fscanf(fp, "%d", &ncategories);
    fscanf(fp, "%d", &nregions);
    fscanf(fp, "%d", &nportals);
    fscanf(fp, "%d", &nlevels);
    for (int i = 0; i < 5; i++) fscanf(fp, "%d", &dummy);
    fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int i = 0; i < 5; i++) fscanf(fp, "%d", &dummy);
  }

  // Fill in house info
//   this->name = (strcmp(name_buffer, "-")) ? strdup(name_buffer) : NULL;
//   this->label = (strcmp(label_buffer, "-")) ? strdup(label_buffer) : NULL;
//   this->bbox = box;

  // Read levels
  for (int i = 0; i < nlevels; i++) {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &dummy);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int j = 0; j < 5; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "L")) { fprintf(stderr, "Error reading level %d\n", i); return 0; }
    // MPLevel *level = new MPLevel();
    // level->position = position;
    // level->label = (strcmp(label_buffer, "-")) ? strdup(label_buffer) : NULL;
    // level->bbox = box;
    // InsertLevel(level);
  }
    
  // Read regions
  for (int i = 0; i < nregions; i++) {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &level_index);
    fscanf(fp, "%d%d", &dummy, &dummy);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    fscanf(fp, "%lf", &height);
    for (int j = 0; j < 4; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "R")) { fprintf(stderr, "Error reading region %d\n", i); return 0; }
    MPRegion *region = new MPRegion();
    // region->position = position;
    region->label = (strcmp(label_buffer, "-")) ? strdup(label_buffer) : NULL;
    // region->bbox = box;
    // region->height = (height > 0) ? height : bbox.ZMax() - position.Z();
    insertRegion(region);
    // if (level_index >= 0) {
    //   MPLevel *level = levels.Kth(level_index);
    //   level->InsertRegion(region);
    // }
  }
    
  // Read portals
  for (int i = 0; i < nportals; i++) {
    int region0_index, region1_index;
    double p0[3], p1[3];
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &region0_index);
    fscanf(fp, "%d", &region1_index);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%lf%lf%lf", &p0[0], &p0[1], &p0[2]);
    fscanf(fp, "%lf%lf%lf", &p1[0], &p1[1], &p1[2]);
    for (int j = 0; j < 4; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "P")) { fprintf(stderr, "Error reading portal %d\n", i); return 0; }
    // MPPortal *portal = new MPPortal();
    // portal->span.Reset(p0, p1);
    // portal->label = (strcmp(label_buffer, "-")) ? strdup(label_buffer) : NULL;
    // InsertPortal(portal);
    // if (region0_index >= 0) {
    //   MPRegion *region0 = regions.Kth(region0_index);
    //   region0->InsertPortal(portal, 0);
    // }
    // if (region1_index >= 0) {
    //   MPRegion *region1 = regions.Kth(region1_index);
    //   region1->InsertPortal(portal, 1);
    // }
  }
    
  // Read surfaces
  for (int i = 0; i < nsurfaces; i++) {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &region_index);
    fscanf(fp, "%d", &dummy);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    fscanf(fp, "%lf%lf%lf", &normal[0], &normal[1], &normal[2]);
    fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int j = 0; j < 5; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "S")) { fprintf(stderr, "Error reading surface %d\n", i); return 0; }
    // MPSurface *surface = new MPSurface();
    // surface->position = position;
    // surface->normal = normal;
    // surface->label = (strcmp(label_buffer, "-")) ? strdup(label_buffer) : NULL;
    // surface->bbox = box;
    // InsertSurface(surface);
    // if (region_index >= 0) {
    //   MPRegion *region = regions.Kth(region_index);
    //   region->InsertSurface(surface);
    // }
  }
    
  // Read vertices
  for (int i = 0; i < nvertices; i++) {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &surface_index);
    fscanf(fp, "%s", label_buffer);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    fscanf(fp, "%lf%lf%lf", &normal[0], &normal[1], &normal[2]);
    for (int j = 0; j < 3; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "V")) { fprintf(stderr, "Error reading vertex %d\n", i); return 0; }
    // MPVertex *vertex = new MPVertex();
    // vertex->position = position;
    // vertex->normal = normal;
    // vertex->label = (strcmp(label_buffer, "-")) ? strdup(label_buffer) : NULL;
    // InsertVertex(vertex);
    // if (surface_index >= 0) {
    //   MPSurface *surface = surfaces.Kth(surface_index);
    //   surface->InsertVertex(vertex);
    // }
  }

  // Read panoramas
  for (int i = 0; i < npanoramas; i++) {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%s", name_buffer);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &region_index);
    fscanf(fp, "%d", &dummy);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    for (int j = 0; j < 5; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "P")) { fprintf(stderr, "Error reading panorama %d\n", i); return 0; }
    MPPanorama *panorama = new MPPanorama();
    panorama->name = (strcmp(name_buffer, "-")) ? strdup(name_buffer) : NULL;
    insertPanorama(panorama);
    if (region_index >= 0) {
      MPRegion *region = regions[region_index];
      region->InsertPanorama(panorama);
    }
  }

  // Read images
  for (int i = 0; i < nimages; i++) {
    double intrinsics[9];
    double extrinsics[16];
    int camera_index, yaw_index, width, height;
    char depth_filename[1024], color_filename[1024];
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &panorama_index);
    fscanf(fp, "%s", name_buffer);
    fscanf(fp, "%d", &camera_index);
    fscanf(fp, "%d", &yaw_index);
    for (int j = 0; j < 16; j++) fscanf(fp, "%lf", &extrinsics[j]);
    for (int j = 0; j < 9; j++) fscanf(fp, "%lf", &intrinsics[j]);
    fscanf(fp, "%d%d", &width, &height);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    for (int j = 0; j < 5; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "I")) { fprintf(stderr, "Error reading image %d\n", i); return 0; }
    sprintf(depth_filename, "%s_d%d_%d.png", name_buffer, camera_index, yaw_index);
    sprintf(color_filename, "%s_i%d_%d.jpg", name_buffer, camera_index, yaw_index);
    MPImage *image = new MPImage();
    image->name = (strcmp(name_buffer, "-")) ? strdup(name_buffer) : NULL;
    image->depth_filename = (strcmp(name_buffer, "-")) ? strdup(depth_filename) : NULL;
    image->color_filename = (strcmp(name_buffer, "-")) ? strdup(color_filename) : NULL;
    image->camera_index = camera_index;
    image->yaw_index = yaw_index;
    // image->rgbd.SetNPixels(width, height);
    // image->rgbd.SetExtrinsics(R4Matrix(extrinsics));
    // image->rgbd.SetIntrinsics(R3Matrix(intrinsics));
    // image->rgbd.SetDepthFilename(depth_filename);
    // image->rgbd.SetColorFilename(color_filename);
    // image->rgbd.SetName(name_buffer);

    std::copy(extrinsics, extrinsics + 16, image->extrinsics);
    std::copy(intrinsics, intrinsics + 9, image->intrinsics);

    image->width = width;
    image->height = height;
    // image->position = position;
    insertImage(image);
    if (panorama_index >= 0) {
      MPPanorama *panorama = panoramas[panorama_index];
      panorama->InsertImage(image);
    }
  }

  // Read categories
  for (int i = 0; i < ncategories; i++) {
    int label_id, mpcat40_id;
    char label_name[1024], mpcat40_name[1024];
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d %s", &label_id, label_name);
    fscanf(fp, "%d %s", &mpcat40_id, mpcat40_name);
    for (int j = 0; j < 5; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "C")) { fprintf(stderr, "Error reading category %d\n", i); return 0; }
    char *label_namep = label_name; while (*label_namep) { if (*label_namep == '#') *label_namep = ' '; label_namep++; }
    char *mpcat40_namep = mpcat40_name; while (*mpcat40_namep) { if (*mpcat40_namep == '#') *mpcat40_namep = ' '; mpcat40_namep++; }
    // MPCategory *category = new MPCategory();
    // category->label_id = label_id;
    // category->mpcat40_id = mpcat40_id;
    // if (strcmp(label_name, "-")) category->label_name = strdup(label_name);
    // if (strcmp(mpcat40_name, "-")) category->mpcat40_name = strdup(mpcat40_name);
    // InsertCategory(category);
  }
    
  // Read objects
  for (int i = 0; i < nobjects; i++) {
    double axis0[3], axis1[3], radius[3];
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &region_index);
    fscanf(fp, "%d", &category_index);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    fscanf(fp, "%lf%lf%lf", &axis0[0], &axis0[1], &axis0[2]);
    fscanf(fp, "%lf%lf%lf", &axis1[0], &axis1[1], &axis1[2]);
    fscanf(fp, "%lf%lf%lf", &radius[0], &radius[1], &radius[2]);
    for (int j = 0; j < 8; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "O")) { fprintf(stderr, "Error reading object %d\n", i); return 0; }
    // MPObject *object = new MPObject();
    // object->position = position;
    // object->obb = R3OrientedBox(position, axis0, axis1, radius[0], radius[1], radius[2]);
    // InsertObject(object);
    // if (region_index >= 0) {
    //   MPRegion *region = regions.Kth(region_index);
    //   region->InsertObject(object);
    // }
    // if (category_index >= 0) {
    //   MPCategory *category = categories.Kth(category_index);
    //   category->InsertObject(object);
    // }
  }
    
  // Read segments
  for (int i = 0; i < nsegments; i++) {
    fscanf(fp, "%s", cmd);
    fscanf(fp, "%d", &house_index);
    fscanf(fp, "%d", &object_index);
    fscanf(fp, "%d", &id);
    fscanf(fp, "%lf", &area);
    fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int j = 0; j < 5; j++) fscanf(fp, "%d", &dummy);
    if (strcmp(cmd, "E")) { fprintf(stderr, "Error reading segment %d\n", i); return 0; }
    // MPSegment *segment = new MPSegment();
    // segment->id = id;
    // segment->area = area;
    // segment->position = position;
    // segment->bbox = box;
    // InsertSegment(segment);
    // if (object_index >= 0) {
    //   MPObject *object = objects.Kth(object_index);
    //   object->InsertSegment(segment);
    // }
  }
    
  // Close file
  fclose(fp);

  // Return success
  return 1;
}

void MP_Parser::insertRegion(MPRegion* region){
    region->house_index = regions.size();
    regions.emplace_back(region);
}

void MP_Parser::insertPanorama(MPPanorama* panorama){
    panorama->region_index = panoramas.size();
    panoramas.emplace_back(panorama);
}

void MP_Parser::insertImage(MPImage* image){
    image->house_index = images.size();
    images.emplace_back(image);
}


