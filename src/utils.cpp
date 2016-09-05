#include <imager.h>
#include "wrappers_cimglist.h"
using namespace Rcpp;
using namespace cimg_library;



// [[Rcpp::export]]
NumericVector load_image(std::string fname) {
  try{
    CId image(fname.c_str());
    return wrap(image);
    }
  catch(CImgException &e){
    forward_exception_to_r(e);
    NumericVector empty;
    return empty; //won't happen
  }
}


// [[Rcpp::export]]
void save_image(NumericVector im, std::string fname) {
  try{
    CId image = as<CId >(im);
    image.save(fname.c_str());
    }
  catch(CImgException &e){
    forward_exception_to_r(e);
  }
  return;
}

//' Split an image along a certain axis (producing a list)
//' 
//' @param im an image 
//' @param axis the axis along which to split (for example 'c')
//' @param nb number of objects to split into. 
//' if nb=-1 (the default) the maximum number of splits is used ie. split(im,"c") produces a list containing all individual colour channels
//' @seealso imappend (the reverse operation)
// [[Rcpp::export]]
List im_split(NumericVector im,char axis,int nb=-1)
{
  try{
    CId img = as<CId >(im);
    CImgList<double> out;
    out = img.get_split(axis,nb);
    return wrap(out);
    }
  catch(CImgException &e){
    forward_exception_to_r(e);
    List empty;
    return empty; //won't happen
  }
}

//' Combine a list of images into a single image 
//' 
//' All images will be concatenated along the x,y,z, or c axis.
//' 
//' @param imlist a list of images (all elements must be of class cimg) 
//' @param axis the axis along which to concatenate (for example 'c')
//' @seealso imsplit (the reverse operation)
//' @export
//' @examples
//' imappend(list(boats,boats),"x") %>% plot
//' imappend(list(boats,boats),"y") %>% plot
//' plyr::rlply(3,imnoise(100,100)) %>% imappend("c") %>% plot
//' boats.gs <- grayscale(boats)
//' plyr::llply(seq(1,5,l=3),function(v) isoblur(boats.gs,v)) %>% imappend("c") %>% plot
// [[Rcpp::export]]
NumericVector imappend(List imlist,char axis)
{
  try{
    CImgList<double> ilist = sharedCImgList(imlist);
    CId out(ilist.get_append(axis));
    //   out.display();
    return wrap(out);
    }
  catch(CImgException &e){
    forward_exception_to_r(e);
    NumericVector empty;
    return empty;
  }
}

//' Pixel-wise evaluation of a CImg expression
//'
//' This function provides experimental support for CImg's "math expression parser", a byte-compiled mini-language. 
//' @param im an image
//' @param expr an expression (as string)
//' @examples
//' imfill(10,10) %>% imeval('x+y') %>% plot
//' # Box filter
//' boxf = "v=0;for(iy=y-3,iy<y+3,iy++,for(ix=x-3,ix< x+3,ix++,v+=i(ix,iy)));v"
//' imeval(boats,boxf) %>% plot
//' # Example by D. Tschumperle: Julia set
//' julia <-  "
//'    zr = -1.2 + 2.4*x/w;
//'    zi = -1.2 + 2.4*y/h;
//'    for (iter = 0, zr^2+zi^2<=4 && iter<256, iter++,
//'      t = zr^2 - zi^2 + 0.5;
//'      (zi *= 2*zr) += 0.2;
//'      zr = t
//'    );
//'    iter"
//' imfill(500,500) %>% imeval(julia) %>% plot
//' @export
// [[Rcpp::export]]
NumericVector imeval(NumericVector im,std::string expr)
{
    CImg<double> img = as<CImg<double> >(im);
    img.fill(expr.c_str(),true);
    return wrap(img);
}

//' Extract a numerical summary from image patches, using CImg's mini-language
//' Experimental feature. 
//' @param im an image
//' @param expr a CImg expression (as a string)
//' @param cx vector of x coordinates for patch centers 
//' @param cy vector of y coordinates for patch centers 
//' @param wx vector of coordinates for patch width 
//' @param wy vector of coordinates for patch height 
//' @examples
//' #Example: median filtering using patch_summary_cimg
//' #Center a patch at each pixel
//' im <- grayscale(boats)
//' patches <- pixel.grid(im)  %>% mutate(w=3,h=3)
//' #Extract patch summary
//' out <- mutate(patches,med=patch_summary_cimg(im,"ic",x,y,w,h))
//' as.cimg(out,v.name="med") %>% plot
//' @export
// [[Rcpp::export]]

NumericVector patch_summary_cimg(NumericVector im,std::string expr,IntegerVector cx,IntegerVector cy,IntegerVector wx,IntegerVector wy)
{
  CId img = as<CId >(im);
  int n = cx.length();
  NumericVector out(n);

  for (int i = 0; i < n; i++)
    {
      out[i] = img.get_crop(cx(i)-wx(i)/2,cy(i)-wy(i)/2,cx(i)+wx(i)/2,cy(i)+wy(i)/2).eval(expr.c_str());
    }
  return out;
}

// Extract a patch summary, fast version
// Modified from original contribution by Martin Roth
// [[Rcpp::export]]
NumericVector extract_fast(NumericVector im,int fun,IntegerVector cx,IntegerVector cy,IntegerVector wx,IntegerVector wy)
{
  CId img = as<CId >(im);
  int n = cx.length();
  NumericVector out(n);
  CId patch;
  
  for (int i = 0; i < n; i++)
  {
    patch = img.get_crop(cx(i)-wx(i)/2,cy(i)-wy(i)/2,cx(i)+wx(i)/2,cy(i)+wy(i)/2);
    switch (fun)
      {
      case 0:
	out[i] = patch.sum();
	break;
      case 1:
	out[i] = patch.mean();
	break;
      case 2:
	out[i] = patch.min();
	break;
      case 3:
	out[i] = patch.max();
	break;
      case 4:
	out[i] = patch.median();
      	break;
      case 5:
	out[i] = patch.variance();
	break;
      case 6:
	out[i] = sqrt(patch.variance());
      }
  }
  return out;
}

//' Extract image patches and return a list
//'
//' Patches are rectangular (cubic) image regions centered at cx,cy (cz) with width wx and height wy (opt. depth wz)
//' WARNINGS: 
//' - values outside of the image region are considered to be 0.
//' - widths and heights should be odd integers (they're rounded up otherwise). 
//' @param im an image
//' @param cx vector of x coordinates for patch centers 
//' @param cy vector of y coordinates for patch centers 
//' @param wx vector of patch widths (or single value)
//' @param wy vector of patch heights (or single value)
//' @return a list of image patches (cimg objects)
//' @export
//' @examples
//' #2 patches of size 5x5 located at (10,10) and (10,20)
//' extract_patches(boats,c(10,10),c(10,20),5,5)
// [[Rcpp::export]]
List extract_patches(NumericVector im,IntegerVector cx,IntegerVector cy,IntegerVector wx,IntegerVector wy)
{
  CId img = as<CId >(im);
  int n = cx.length();
  List out(n);
  bool rep = false;
  if (cx.length() != cy.length())
    {
      stop("cx and cy must have equal length");
    }
  if (wx.length() != wy.length())
    {
      stop("wx and wy must have equal length");
    }
  if (wx.length() == 1)
    {
      rep = true;
    }
  cx = cx - 1;
  cy = cy - 1;
  for (int i = 0; i < n; i++)
    {
      if (rep)
	{
	  out[i] = wrap(img.get_crop(cx(i)-wx(0)/2,cy(i)-wy(0)/2,cx(i)+wx(0)/2,cy(i)+wy(0)/2)); 
	}
      else
	{
	  out[i] = wrap(img.get_crop(cx(i)-wx(i)/2,cy(i)-wy(i)/2,cx(i)+wx(i)/2,cy(i)+wy(i)/2)); 
	}
    }
  return out;
}

//' @param cz vector of z coordinates for patch centers 
//' @param wz vector of coordinates for patch depth
//' @describeIn extract_patches Extract 3D patches
//' @export
// [[Rcpp::export]]
List extract_patches3D(NumericVector im,IntegerVector cx,IntegerVector cy,IntegerVector cz,IntegerVector wx,IntegerVector wy,IntegerVector wz)
{
  CId img = as<CId >(im);
  int n = cx.length();
  List out(n);
  bool rep = false;
  if ((cx.length() != cy.length()) or (cx.length() != cz.length()) or (cy.length() != cz.length()))
    {
      stop("cx, cy and cz must have equal length");
    }
  if ((wx.length() != wy.length()) or (wx.length() != wz.length()) or (wy.length() != wz.length()))
    {
      stop("wx, wy and wz must have equal length");
    }
  if (wx.length() == 1)
    {
      rep = true;
    }
  for (int i = 0; i < n; i++)
    {
      if (rep)
	{
	  out[i] = img.get_crop(cx(i)-wx(0)/2,cy(i)-wy(0)/2,cz(i)-wz(0)/2,cx(i)+wx(0)/2,cy(i)+wy(0)/2,cz(i)+wz(0)/2);
	}
      else
	{
	  out[i] = img.get_crop(cx(i)-wx(i)/2,cy(i)-wy(i)/2,cz(i)-wz(i)/2,cx(i)+wx(i)/2,cy(i)+wy(i)/2,cz(i)+wz(i)/2);
	}
    }
  return out;
}

// [[Rcpp::export]]
NumericVector draw_image(NumericVector im,NumericVector sprite,int x=0,int y = 0, int z = 0,float opacity = 1)
{
  CId img = as<CId >(im);

  try{
    CId spr = as<CId >(sprite);
    img.draw_image(x,y,z,spr,opacity);
    }
  catch(CImgException &e){
    forward_exception_to_r(e);
    
  }
  return wrap(img);
}


// [[Rcpp::export]]
List do_patchmatch(NumericVector im1,NumericVector im2,
			  unsigned int patch_width,
			  unsigned int patch_height,
			  unsigned int patch_depth,
			  unsigned int nb_iterations,
			  unsigned int nb_randoms,
			  NumericVector guide)
{
  try{
    CId img1 = as<CId >(im1);
    CId img2 = as<CId >(im2);
    CId g = as<CId >(guide);
    CId mscore(img1,"xyzc");
    CImg<int> out = img1.patchmatch(img2,patch_width,patch_height,patch_depth,
				    nb_iterations,nb_randoms,g,mscore);
    CId outfl(out);
    return List::create(_["warp"] = wrap(outfl),_["score"] = wrap(mscore));
    }
  catch(CImgException &e){
    forward_exception_to_r(e);
    List empty;
    return empty; //won't happen
  }

}



// Check that coordinates are all in image (indexing from 1)
// [[Rcpp::export]]
LogicalVector checkcoords(IntegerVector x,IntegerVector y,IntegerVector z,IntegerVector c,IntegerVector d)
{
  int n = x.length();
  LogicalVector out(n);
  for (int i = 0; i < n; i++)
    {
      if ((x[i] < 1) or (x[i] > d[0]) or (y[i] < 1) or (y[i] > d[1]) or (z[i] < 1) or (z[i] > d[2]) or (c[i] < 1) or (c[i] > d[3]))
	{
	  out[i] = false;
	}
      else
	{
	  out[i] = true;
	}
    }
  return out;
}


// [[Rcpp::export]]
int cimg_omp()
{
  return cimg::openmp_mode();
}

// [[Rcpp::export]]
int set_cimg_omp(int mode)
{
  return cimg::openmp_mode(mode);
}

// [[Rcpp::export]]
bool has_omp()
{
#ifdef cimg_use_openmp
  return true;
#else
  return false;
#endif
}
