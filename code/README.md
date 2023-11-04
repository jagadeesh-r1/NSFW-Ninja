Project code goes here.


# GCP Detect explicit content (SafeSearch) Cloud Vision API 

## Problems
<strike>
1. There are 100K images in my local instance, sync processing these 100K images will take a lot of time 
2. The Vision API now supports offline asynchronous batch image annotation for all features. This asynchronous request supports up to 2000 image files and returns response JSON files that are stored in your Cloud Storage bucket. But Cost for this operation should be kept in mind
</strike>