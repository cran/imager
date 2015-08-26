#define cimg_use_fftw3
#include <imager.h>
using namespace Rcpp;
using namespace cimg_library;

//' Apply recursive Deriche filter.
//'
//' @param im an image
//' @param sigma Standard deviation of the filter.
//' @param order Order of the filter. Can be <tt>{ 0=smooth-filter | 1=1st-derivative | 2=2nd-derivative }</tt>.
//' @param axis Axis along which the filter is computed. Can be <tt>{ 'x' | 'y' | 'z' | 'c' }</tt>.
//' @param boundary_conditions Boundary conditions. Can be <tt>{ 0=dirichlet | 1=neumann }</tt>.
//' @export
//' @examples
//' deriche(boats,sigma=2,order=0) %>% plot("Zeroth-order Deriche along x")
//' deriche(boats,sigma=2,order=1) %>% plot("First-order Deriche along x")
//' deriche(boats,sigma=2,order=1) %>% plot("Second-order Deriche along x")
//' deriche(boats,sigma=2,order=1,axis="y") %>% plot("Second-order Deriche along y")
// [[Rcpp::export]]
NumericVector deriche(NumericVector im,float sigma,int order=0,char axis = 'x',bool boundary_conditions=0)
{
  CId img = as<CId >(im);
  img.deriche(sigma,order,axis,boundary_conditions);
  return wrap(img);
}


//' Van Vliet recursive Gaussian filter.
//'
//'       From: I.T. Young, L.J. van Vliet, M. van Ginkel, Recursive Gabor filtering.
//'       IEEE Trans. Sig. Proc., vol. 50, pp. 2799-2805, 2002.
//'       (this is an improvement over Young-Van Vliet, Sig. Proc. 44, 1995)
//'
//'       Boundary conditions (only for order 0) using Triggs matrix, from
//'       B. Triggs and M. Sdika. Boundary conditions for Young-van Vliet
//'       recursive filtering. IEEE Trans. Signal Processing,
//'       vol. 54, pp. 2365-2367, 2006.
//'
//'       @param im an image
//'       @param sigma standard deviation of the Gaussian filter
//'       @param order the order of the filter 0,1,2,3
//'       @param axis  Axis along which the filter is computed. Can be <tt>{ 'x' | 'y' | 'z' | 'c' }</tt>.
//'       @param boundary_conditions Boundary conditions. Can be <tt>{ 0=dirichlet | 1=neumann }</tt>.
//'       (Dirichlet boundary condition has a strange behavior)
//' @examples
//' vanvliet(boats,sigma=2,order=0) %>% plot("Zeroth-order Vanvliet along x")
//' vanvliet(boats,sigma=2,order=1) %>% plot("First-order Vanvliet along x")
//' vanvliet(boats,sigma=2,order=1) %>% plot("Second-order Vanvliet along x")
//' vanvliet(boats,sigma=2,order=1,axis="y") %>% plot("Second-order Vanvliet along y")
//' @export
// [[Rcpp::export]]
NumericVector vanvliet(NumericVector im,float sigma,int order=0,char axis = 'x',bool boundary_conditions=0)
{
  CId img = as<CId >(im);
  img.vanvliet(sigma,order,axis,boundary_conditions);
  return wrap(img);
}


//' Blur image isotropically.
//' @param im an image
//' @param sigma Standard deviation of the blur.
//' @param boundary_conditions Boundary conditions. Can be <tt>{ 0=dirichlet | 1=neumann }
//' @param gaussian Use a Gaussian filter (default FALSE). Default: O-order Deriche filter.
//' @seealso deriche
//' @export
//' @examples
//' isoblur(boats,3) %>% plot(main="Isotropic blur, sigma=3")
//' isoblur(boats,3) %>% plot(main="Isotropic blur, sigma=10")
//' @seealso medianblur
// [[Rcpp::export]]
NumericVector isoblur(NumericVector im,float sigma,bool boundary_conditions=true,bool gaussian=false) {
  CId img = as< CId >(im);
  img.blur(sigma,boundary_conditions,gaussian);
  return wrap(img);
}


//' Blur image with the median filter.
//'    
//' @param im an image
//'  @param n Size of the median filter.
//'  @param threshold Threshold used to discard pixels too far from the current pixel value in the median computation. Can be used for edge-preserving smoothing. 
//' @export
//' @examples
//' medianblur(boats,5,Inf) %>% plot(main="Median blur, 5 pixels")
//' medianblur(boats,10,Inf) %>% plot(main="Median blur, 10 pixels")
//' medianblur(boats,10,8) %>% plot(main="Median blur, 10 pixels, threshold = 8")
//' @seealso isoblur, boxblur
// [[Rcpp::export]]
NumericVector medianblur(NumericVector im,int n, float threshold) {
  CId img = as<CId >(im);
  img.blur_median(n,threshold);
  return wrap(img);
}

//' Blur image with a box filter (square window)
//' @param im an image
//' @param sigma Size of the box window.
//' @param boundary_conditions Boundary conditions. FALSE: Dirichlet TRUE: Neumann.
//' @seealso deriche(), vanvliet().
//' @examples
//' boxblur(boats,5) %>% plot(main="Dirichlet boundary")
//' boxblur(boats,5,TRUE) %>% plot(main="Neumann boundary")
//' @export
// [[Rcpp::export]]
NumericVector boxblur(NumericVector im,float sigma,bool boundary_conditions=true) {
  CId img = as<CId >(im);
  img.blur_box(sigma,boundary_conditions);
  return wrap(img);
}


//' Blur image with a box filter.
//'
//' This is a recursive algorithm, not depending on the values of the box kernel size.
//'
//' @param im an image
//'       @param sx Size of the box window, along the X-axis.
//'       @param sy Size of the box window, along the Y-axis.
//'       @param boundary_conditions Boundary conditions. Can be <tt>{ false=dirichlet | true=neumann }</tt>.
//'       @seealso blur().
//'
//' @export
//' @examples
//' boxblur_xy(boats,20,5) %>% plot(main="Anisotropic blur")
// [[Rcpp::export]]
NumericVector boxblur_xy(NumericVector im,float sx,float sy,bool boundary_conditions=true) {
  CId img = as<CId >(im);
  img.blur_box(sx,sy,0,boundary_conditions);
  return wrap(img);
}

//' Correlation of image by filter
//'
//'  The correlation of image im by filter flt is defined as:
//'  \eqn{res(x,y,z) = sum_{i,j,k} im(x + i,y + j,z + k)*flt(i,j,k).}
//'
//'       @param im an image
//'       @param filter = the correlation kernel.
//'       @param boundary_conditions = the border condition type (0=zero, 1=dirichlet)
//'       @param normalise  = normalise filter (default FALSE)
//'      
//'
//' @export
//' @examples
//' #Edge filter
//' filter <- as.cimg(function(x,y) sign(x-5),10,10) 
//' layout(t(1:2))
//' #Convolution vs. correlation 
//' correlate(boats,filter) %>% plot(main="Correlation")
//' convolve(boats,filter) %>% plot(main="Convolution")
// [[Rcpp::export]]
NumericVector correlate(NumericVector im,NumericVector filter, bool boundary_conditions=true,bool normalise = false) {
  CId img = as<CId >(im);
  CId flt = as<CId >(filter);
  img.correlate(flt,boundary_conditions,normalise);
  return wrap(img);
}


//' Convolve image by filter.
//'
//'      The result  res of the convolution of an image img by filter flt is defined to be:
//'       \eqn{res(x,y,z) = sum_{i,j,k} img(x-i,y-j,z-k)*flt(i,j,k)}
//'
//'       @param im an image
//'       @param filter a filter (another image)
//'       @param boundary_conditions = the border condition type (0=zero, 1=dirichlet)
//'       @param normalise = normalise filter (default FALSE)
//'
//'
//' @export
//' @seealso correlate
//' @examples
//' #Edge filter
//' filter <- as.cimg(function(x,y) sign(x-5),10,10) 
//' layout(t(1:2))
//' #Convolution vs. correlation 
//' correlate(boats,filter) %>% plot(main="Correlation")
//' convolve(boats,filter) %>% plot(main="Convolution")
// [[Rcpp::export]]
NumericVector convolve(NumericVector im,NumericVector filter, bool boundary_conditions=true,bool normalise = false) {
  CId img = as<CId >(im);
  CId flt = as<CId >(filter);
  img.convolve(flt,boundary_conditions,normalise);
  return wrap(img);
}


//' Sharpen image.
//'
//'       @param im an image
//'       @param amplitude Sharpening amplitude
//'       @param sharpen_type Select sharpening method. Can be <tt>{ false=inverse diffusion | true=shock filters }</tt>.
//'       @param edge Edge threshold (shock filters only).
//'       @param alpha Gradient smoothness (shock filters only).
//'       @param sigma Tensor smoothness (shock filters only).
//'
//' @export
//' @examples
//' layout(t(1:2))
//' plot(boats,main="Original")
//' imsharpen(boats,150)  %>% plot(main="Sharpened")
//' 
// [[Rcpp::export]]
NumericVector imsharpen(NumericVector im,float amplitude,
		bool sharpen_type = false,float edge = 1,
		float alpha = 0,float sigma = 0)
 {
   CId img = as<CId >(im);
   img.sharpen(amplitude,sharpen_type,edge,alpha,sigma);
   return wrap(img);
}

//' Compute image gradient.
//'
//' @param im an image
//' @param axes Axes considered for the gradient computation, as a C-string (e.g "xy").
//' @param scheme = Numerical scheme used for the gradient computation:
//'       1 = Backward finite differences
//'       0 = Centered finite differences
//'       1 = Forward finite differences
//'       2 = Using Sobel masks
//'       3 = Using rotation invariant masks
//'       4 = Using Deriche recursive filter.
//'       5 = Using Van Vliet recursive filter.
//' @return a list of images (corresponding to the different directions)
//' @export
//' @seealso imgradient
// [[Rcpp::export]]
List get_gradient(NumericVector im,std::string axes = "",int scheme=3)
{
   CId img = as<CId >(im);
   CImgList<double> grad = img.get_gradient(axes.c_str(),scheme);
   return wrap(grad);
}

//' Return image hessian.
//' @param im an image
//' @param axes Axes considered for the hessian computation, as a character string (e.g "xy").
// [[Rcpp::export]]
List get_hessian(NumericVector im,std::string axes = "")
{
   CId img = as<CId >(im);
   CImgList<double> hess = img.get_hessian(axes.c_str());
   //CId out(im);
   //out = img.get_hessian(axes.c_str());
   return wrap(hess);
}

//' Compute field of diffusion tensors for edge-preserving smoothing.
//'
//'       @param im an image
//'       @param sharpness Sharpness
//'       @param anisotropy Anisotropy
//'       @param alpha Standard deviation of the gradient blur.
//'       @param sigma Standard deviation of the structure tensor blur.
//'       @param is_sqrt Tells if the square root of the tensor field is computed instead.
//' @export
// [[Rcpp::export]]
NumericVector diffusion_tensors(NumericVector im,
				float sharpness = 0.7,	float anisotropy = 0.6,
				float alpha = 0.6,float sigma = 1.1,
				bool is_sqrt = false) 	
{
  CId img = as<CId >(im);
  img.diffusion_tensors(sharpness,anisotropy,alpha,sigma,is_sqrt);
  return wrap(img);
}

//' Compute Haar multiscale wavelet transform.
//'
//'       @param im an image
//'       @param inverse Compute inverse transform (default FALSE)
//'       @param nb_scales Number of scales used for the transform.
//'
//' @export
//' @examples
//' #Image compression: set small Haar coefficients to 0
//' hr <- haar(boats,nb=3) 
//' mask.low <- threshold(abs(hr),"75%")
//' mask.high <- threshold(abs(hr),"95%")
//' haar(hr*mask.low,inverse=TRUE,nb=3) %>% plot(main="75% compression")
//' haar(hr*mask.high,inverse=TRUE,nb=3) %>% plot(main="95% compression")
// [[Rcpp::export]]
NumericVector haar(NumericVector im,bool inverse=false,int nb_scales=1) {
  CId img = as<CId >(im);
  img.haar(inverse,nb_scales);
  return wrap(img);
}

// [[Rcpp::export]]
List FFT_complex(NumericVector real,NumericVector imag,bool inverse=false,int nb_threads=0) {
  CId rl = as<CId >(real);
  CId img = as<CId >(imag);
  rl.FFT(rl,img,inverse,nb_threads);
  return List::create(_["real"] = wrap(rl),_["imag"] = wrap(img));
}

// [[Rcpp::export]]
List FFT_realim(NumericVector real,bool inverse=false,int nb_threads=0) {
  CId rl = as<CId >(real);
  CId im(rl,"xyzc",0);
  rl.FFT(rl,im,inverse,nb_threads);
  return List::create(_["real"] = wrap(rl),_["imag"] = wrap(im));
}

// [[Rcpp::export]]
NumericVector FFT_realout(NumericVector real,NumericVector imag,bool inverse=false,int nb_threads=0) {
  CId rl = as<CId >(real);
  CId img = as<CId >(imag);
  rl.FFT(rl,img,inverse,nb_threads);
  return wrap(rl);
}


//' Estimate displacement field between two images.
//'
//' @param sourceIm Reference image.
//' @param destIm Reference image.
//' @param smoothness Smoothness of estimated displacement field.
//' @param precision Precision required for algorithm convergence.
//' @param nb_scales Number of scales used to estimate the displacement field.
//' @param iteration_max Maximum number of iterations allowed for one scale.
//' @param is_backward If false, match I2(X + U(X)) = I1(X), else match I2(X) = I1(X - U(X)).
//' @export
// [[Rcpp::export]]
NumericVector displacement(NumericVector sourceIm,NumericVector destIm,float smoothness=0.1, float precision=5.0,unsigned int nb_scales=0, unsigned int iteration_max=10000,bool is_backward=false)
{
   CId src = as<CId >(sourceIm);
   CId dst = as<CId >(destIm);
   CId out(src,false);
   out.displacement(dst,smoothness,precision,nb_scales,iteration_max,is_backward);
   return wrap(out);
}


//' Blur image anisotropically, in an edge-preserving way.
//' @param im an image
//' @param amplitude Amplitude of the smoothing.
//' @param sharpness Sharpness.
//' @param anisotropy Anisotropy.
//' @param alpha Standard deviation of the gradient blur.
//' @param sigma Standard deviation of the structure tensor blur.
//' @param dl Spatial discretization.
//' @param da Angular discretization.
//' @param gauss_prec Precision of the diffusion process.
//' @param interpolation_type Interpolation scheme.
//'  Can be 0=nearest-neighbor | 1=linear | 2=Runge-Kutta 
//' @param is_fast_approx Determines if a fast approximation of the gaussian function is used or not.
//' @export
//' @examples
//' im <- load.image(system.file('extdata/Leonardo_Birds.jpg',package='imager'))
//' im.noisy <- (im + 80*rnorm(prod(dim(im)))) 
//' blur_anisotropic(im.noisy,ampl=1e4,sharp=1) %>% plot
// [[Rcpp::export]]
NumericVector blur_anisotropic(NumericVector im, float amplitude,  float sharpness=0.7,  float anisotropy=0.6,float alpha=0.6,  float sigma=1.1,  float dl=0.8,  float da=30,
                               float gauss_prec=2,  unsigned int interpolation_type=0,
                               bool is_fast_approx=true) {
  CId img = as<CId >(im);
  img.blur_anisotropic(amplitude,sharpness,anisotropy,alpha,sigma,dl,da,gauss_prec,interpolation_type,is_fast_approx);
  return wrap(img);
}

// [[Rcpp::export]]
NumericVector periodic_part(NumericVector im)
{
  CId img = as<CId >(im);
  CId D(img,"xyzc",0);
  if (img.spectrum() > 1 or img.depth() > 1)
    {
      stop("This function works only on 2D grayscale images");
    }
  int w = img.width(),h = img.height();
  int x = 0,y=0;
  double f = 0;
  //Compute D matrix (differences across boundaries)
  for (x = 0; x < w; x++)
    {
      f = img(x,0) - img(x,h-1);
      D(x,0) = -f;
      D(x,h-1) = f;
    }
  x = 0;
  for (y = 0; y < h; y++)
    {
      f = img(0,y) - img(w-1,y);
      D(0,y) -= f;
      D(w-1,y) += f;
    }
  //Compute FFT of D 
  CId impart(D,"xyzc",0);
  CId realpart(D);
  double weight=0;
  D.FFT(realpart,impart);
  cimg_forXY(D,x,y)
    {
      weight = 2*cos(2*cimg::PI*x/w)+2*cos(2*cimg::PI*y/h) - 4;
      impart(x,y) /= weight;
      realpart(x,y) /= weight;
    }
  realpart(0,0) = 0;
  impart(0,0) = 0;
  //Inverse FFT
  D.FFT(realpart,impart,true);
  //Take out non-periodic part from the original image
  img -= realpart;
  return wrap(img);
}
