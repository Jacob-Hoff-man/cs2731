# CS-2731 Final Project Code Instructions

NOTE: the course material used in this project has not been uploaded to GitHub.

## Preprocessing

To preprocess the slides in the data folder to the annotation form, execute `fmt_for_annotation.py`. Be warned, this will remove the labels from the data in `json_data`. You can recover these if done accidentally, by executing `fmt_change_back.py`, which will take the labels in the `json_pair_slides` fmt and put them in the `json_slides` fmt.

Next, once you have finished annotating the items in `json_data`, or would like to annotate them in the `json_pair_slides` form, execute `fmt_change.py`. 

Finally, when you are ready to get the final dataset (located in `final_slides_json`), execute `fmt_change_to_final.py`.

## Distant Labelling

To label the data you have not labeled using distant labelling, simply move the data from `final_slides_json` to `final_slides_json_distant`. Move the remaining slides to `final_slides_json_manual`, so that they can be used to test the data.

Then run `distant_label.py`

Once it is done, zip these folders, and move onto the next step.

## Testing the Model

To run the model, please see Google Colab notebook file.