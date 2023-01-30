#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# <h1>SuperMart Product Recommendation App</h1>
# <h4>Cedric Green - Computer Science Capstone WGU<h4>
# <p> Product Recommendation Interface for SuperMart. The script that follows represents the main visual of what the interface is going to accomplish. The first process is obtaining an item of interest from the user. Since this is content based filtering, we do not understand at the moment what the user likes. To kickstart our machine learning application our prescriptive method will require initial input of what the user 'might' be interested in. From here we can compare the similarities using machine learning to offer the top 10 items within our product list that will align closely with the user's interests. </p>

# In[ ]:


from IPython.display import display, HTML
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import time

df = pd.read_csv("flipkart.csv")

# Title window
print("SuperMart Product Recommendation Interface\n------------------------------------------------")

# Data Preprocessing

features = ['product_name', 'description', 'brand','product_category_tree', 'overall_rating']

for feature in features:
	df[feature] = df[feature].fillna('')
    
#function  make single feature

def feat_comb(row):
    try:
        if row['overall_rating'] == 'No rating available':
            row['overall_rating'] = '0'
            
        return row['product_name'] + " " + row['brand'] + " " + row['product_category_tree'][2 : row['product_category_tree'].index('>')] + " " + row['overall_rating']
    except:
         if row['overall_rating'] == 'No rating available':
            row['overall_rating'] = '0'
            
         return row['product_name'] + " " + row['brand'] + " " + row['product_category_tree'] + " " + row['overall_rating']


	
df['combined'] = df.apply(feat_comb, axis = 1)


#extract features
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined'])

cosine_sim = cosine_similarity(count_matrix)

user_entry = "initial"

while(user_entry != "Quit"):
    user_entry = input('Please enter quit to exit OR enter a name of an item you may be interested in and we will provide ' + 
                   'SuperMart product recommendations for\nyou (example -> Boots): ')
    user_entry = user_entry.replace(user_entry[0], str.upper(user_entry[0]), 1)
    
    if(user_entry == "Quit"):
        break

    def get_index_from_entry(entry):
        
        print("\nFinding items similar to: " + entry)
    
        for iter in df['product_name']:
            if entry in iter:
                return df.index[df['product_name'] == iter].values[0]   
	

    u_index = get_index_from_entry(user_entry)

    if u_index == None:
        print(f"Unable to recommend using {user_entry}. No similar items exist. Please try again with something more common.")
    else:
    

        similar_products = list(enumerate(cosine_sim[u_index]))

        sorted_similar_products = sorted(similar_products, key=lambda x:x[1], reverse=True)

        def get_name_from_index(index):
            return df[df.index == index]['product_name'].values[0]

        print(f"\n\nTop 10 Product Recommendations for {user_entry}\n")
    
        plot_points = []
    
        i=0
        for prod in sorted_similar_products:
            print(f"Similarity % {round(prod[1], 2)}  -> {i + 1}.) {get_name_from_index(prod[0])}")
            plot_points.append(round(prod[1], 2))
            i=i+1
            if i>9:
                break
        
        time.sleep(5)
        
        plot_points_orig = map(lambda x: x * 100, plot_points)
        plot_points = [*set(plot_points)]
    
        #now that we have the set of plot points iterate through and calculate 
        final_buckets = [0, 1, 0, 0]
        a_labels = ["Not Very Accurate < 30%", "Maybe Accurate 30 - 39% ", "Accurate 40 - 69% ", "High Acccuracy > 70%"]
    
        for x in range(len(plot_points)):
            if plot_points[x]< 30/100:
                final_buckets[0] += 1
            elif plot_points[x] >= 30/100 and plot_points[x] < 40/100:
                final_buckets[1] += 1
            elif plot_points[x] >= 40/100 and plot_points[x] < 70/100:
                final_buckets[2]+= 1
            else:
                final_buckets[3] += 1
    
        print('\n\n\t\t\t\tDONUT CHART VISUAL')
    
        plt.pie(final_buckets, labels=a_labels)
    
        # add a circle at the center to transform it in a donut chart
        my_circle=plt.Circle( (0,0), 0.6, color='white')
        p=plt.gcf()
        p.gca().add_artist(my_circle)

        plt.show()
    
        plot_points_orig = list(plot_points_orig)
    
        b_labels = ['Low Accuracy', 'Mid Accuracy', 'High Accuracy', 'Greatest Accuracy']
        
        time.sleep(5)
        
        #Bar chart representations
        print('\n\t\t\t\tBAR CHART VISUAL')
        plt.bar(b_labels, final_buckets)
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Occurences (Unique)')
        plt.show()
        
        time.sleep(10)
        
        print('\n\t\t\t\tHISTOGRAM VISUAL')
        #Histogram chart representations
        plt.hist(plot_points_orig)
        plt.xlabel('Similarity %')
        plt.ylabel('Frequency')
        plt.show()
        
        time.sleep(5)
        
          # Descriptive Method
        print("\n\n\nDescriptive Method\n----------------------\n")
    
        print("Understanding similarities in terms of their similarity rating yields the following combined points of analysis.")
        print()
    
        print("Data in our dataset that matches closest to the keyword entered by user\nhas the following combined qualities\n")
        print("*Note* Keyword can represent a brand, or an item component which is highly useful within our domain\n we combined them all into a single feature.")

        display(HTML(pd.DataFrame(df.iloc[[u_index]]['combined']).to_html()))
    
        #comparing to the top 10 results
        print("\n\nA look at the top 10 similarity results...")
        i=0
        for prod in sorted_similar_products:
            display(HTML(pd.DataFrame(df.iloc[[prod[0]]]['combined']).to_html()))
            i=i+1
            if i>9:
                break
    
        print("\nGiven the features provided we can accurately describe what we are seeking as input for our" +
             " machine learning model for further processing. As you can see the descriptive properties outlined in" +
             " the data will help us\nsolve the business problem at hand.")
   
    
        time.sleep(5)
    
        # Prescriptive Method
        print("\nPrescriptive Method\n--------------------------\n")
        print("Our prescriptive method is ultimately illustrated in the main visuals above. We take in a keyword and then\n" +
             "our machine learning model is able to efficiently find the cosine similarity of the results and categorize them\n" +
             "in a way that the items with the highest similarity factor are selected and the lowest factors are not applied.\n" +
             "No matter how much new data is added our model can adjust automatically and always predict similar interests and\nfind the closest scored matches.")
        print("One such example of the predictive method at work will be reflected in the DONUT CHART ABOVE. We have 4\ncategories in which" +
              " to train our supervised model based off. If results are reflected as not satisfactory, we can\nthen adjust the matrix" +
             " and tune the model to our liking until satisfactory results are displayed.")
    
        print(f'\nTotal size of dataset: {len(df)}\n')
    
        print("\nOur data model was supplied a sample size of 20,000 different products and corresponding data for the training\nphase. After getting" +
             "an initial positive result we are confident that no further training is\nnecessary at this time. If SuperMart wishes, " +
             "more data can be obtained and the model will be\nfurther refined in the future.")
    
        print("\n\n**NOTE** Please SCROLL UP to see complete in-depth detail layout.")
     
print("\nThanks for trying SuperMart Product Recommender!")


# The Descriptive models shown above lets us know how the model is performing at any given time. The data outlined is organized to show us the accuracy of the machine learning algorithm. The first two results visualizations classify and categorize the similarity rankings of the final result across 4 containers. The final visualization is a histogram representing the results data distribution. If results are not satisfactory for business performance we can easily extend further training on the model to optimize the results or fine tune the model in one direction or another. Having the descriptive performance display will allow us to determine strengths and weaknesses as we work to increase sales. Following the descriptive models and visualizations, a descriptive method was provided. Information that describes the features we found within the data that describe what information we are seeking was extracted from the complete dataset. Following the descriptive method we reveal the prescriptive method.
