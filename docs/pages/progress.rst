
===========================
Weekly Progress
===========================


---------------------------------
December 2025
---------------------------------

............
 11 Dec 2025
............

 **Results** 

- Developed a fitting object that reads an hkl file, bins spots based on
  resolution and then performs the fitting (as in the original paper). The
  fitter then performs a scaling correction on the original data. The only 
  problem is that the corrected datasets have lower :math:`R_1` when processed
  with :code:`shelxt`.

.. list-table::
   :widths: 33 33 33
   :align: center

   * - .. image:: ../figs/fit_by_res.png
          :class: zoomable
          :alt: First
     - .. image:: ../figs/replaced_fc.png
          :class: zoomable
          :alt: Second
     - .. image:: ../figs/r1_correction.png
          :class: zoomable
          :alt: Third

   * - Fitting the intensity correction by resolution.
     - Original :math:`F_{\rm o}` replaced with :math:`F_c`.
     - Correction in :code:`shelxt` :math:`R_1` for datasets with intensities computed directly from Gemmi.

- Replaced the original data for Paracetamol with scaled :math:`|F_c|^2`
  (computed using Gemmi) and compared the resulting :math:`R_1` factors
  obtained by processing with :code:`shelxt`. In the majority of cases, the
  new R-factors are lower than the original ones.
- Trained a Gradient boosted regressor on a single dataset (input parameters:
  H, K, L, intensity, sigma,  image index, resolution, global scaling 
  parameter). 

.. list-table::
   :widths: 33 33 33
   :align: center

   * - .. image:: ../figs/compare_with_fitted.png
          :class: zoomable
          :alt: First
     - .. image:: ../figs/0152_Fc_vs_Fpred.png
          :class: zoomable
          :alt: Second
     - 

   * - Computed :math:`F_c` on the trained data
     - Computed :math:`F_c` on the new data.
     - 


**Discussion** 

- Find the original data for the J. P. Abrahams paper and try to reproduce
  their results to make sure our processing pipeline is correct.
- Compute :math:`R_1` per image to identify images that are highly impacted by
  dynamical effects.
- Retrain the gradient boosted regressor to include more information about
  the spot environment (e.g. maximal and minimal intensity on an image,
  average intensity on an image, miller indices of the neighbouring spots).
  Train on more datasets.
- Read about PointNet Architecture. 

............
 18 Dec 2025
............

 **Results** 
 


 **Discussion** 


.. Commented (How to make image grid)
    .. list-table::
       :widths: 33 33 33
       :align: center

       * - .. image:: ../figs/0152_Fc_vs_Fpred.png
              :class: zoomable
              :alt: First
         - .. image:: ../figs/0152_Fc_vs_Fpred.png
              :class: zoomable
              :alt: Second
         - 

       * - First caption
         - Second caption
         - 

.. Standard standalone figure environment
   .. figure:: ../figs/0152_Fc_vs_Fpred.png
      :alt: Beam position computed using maximum method 
      :class: zoomable
      :width: 33%

      Test image

.. Labeled Math equation (Reference with :eq:`abc`)
    .. math::
       :label: abc

       e^i + 1 = 0 e^i + 1 = 0 e^i + 1 = 0 e^i + 1 = 0 e^i + 1 = 0 
