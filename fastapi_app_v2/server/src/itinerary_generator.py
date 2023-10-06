import datetime
import json
import logging
import os
import requests
import re

from yelp_restaurants import main

from google.cloud import storage
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms import GooglePalm, OpenAI

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)


class ItineraryGenerator:

    def __init__(self):
        self.log_bucket_name = os.getenv("BUCKET_NAME")
        self.feedback_bucket_name = os.getenv("FEEDBACK_BUCKET_NAME")
        self.storage_client = storage.Client()
        self.default_template = self.load_prompt()
        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=self.default_template
        )
        self.selected_llm = None
        self.user_query_template = None
        self.generated_itinerary = None

    def log_llm_response(self, llm, query, itinerary):
        self.selected_llm = llm
        self.user_query_template = query
        self.generated_itinerary = itinerary
        self._upload_to_bucket(self.log_bucket_name, {"id": self._get_unique_id(), "query": query, "llm": llm, "itinerary": itinerary})

    def user_feedback(self, rating, feedback):
        llm, query, itinerary = self.selected_llm, self.user_query_template, self.generated_itinerary
        self._upload_to_bucket(self.feedback_bucket_name,{
            "id": self._get_unique_id(),
            "user_query": query,
            "LLM": llm,
            "itinerary": itinerary,
            "user_rating": rating,
            "user_feedback": feedback
        })

    def _upload_to_bucket(self, bucket_name, data):
        data_str = json.dumps(data)
        bucket = self.storage_client.get_bucket(bucket_name)
        blob_name = f"log_{self._get_unique_id()}_json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data_str)

    @staticmethod
    def _get_unique_id():
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    
    @staticmethod
    def load_prompt():
        """
        Define the prompt template for the itinerary planning.
        """
        return """
            You are an expert intelligent AI itinerary planner with extensive knowledge of places worldwide. Your goal is to plan an optimized itinerary for the user based on their specific interests and preferences, geographical proximity, and efficient routes to minimize travel time. To achieve this, follow these instructions:

            1. Suggest atleast 3 activities per day. Each activity should include the name of the place, a brief description, estimated cost, and time to reach the place.
            
            2. Generate a well-structured itinerary including day-to-day activities, timings to visit each location, and estimated costs for the user's reference.

            2. Take into account factors such as geographical proximity between destinations, transportation options, and other logistical considerations when planning the route.
            
            By following these guidelines, you will create a comprehensive and optimized itinerary that meets the user's expectations while ensuring minimal travel time.

            Current conversation:
            {history}
            Human: {input}
            AI:"""
    
    @staticmethod
    def load_itinerary_template_json(
            destination, budget, arrival_date, departure_date, start_time, end_time, additional_info, restaurants
    ):
        
        query = f"""
            I am planning a trip for you from {arrival_date} to {departure_date} to {destination} with a budget of ${budget}. Start the itinerary each day from {start_time} to {end_time}. Consider additional information regarding {additional_info}, if provided.
        """
        
        template = f"""{query}. 
    Consider budget, timings and requirements. Include estimated cost for each activity.
    Use this restaurants list {restaurants} if needed or suggest by yourself. 
    Structure the itinerary as follows:
    {{"Name":"name of the trip", "description":"description of the entire trip", "budget":"budget of the entire thing", "data": [{{"day":1, "day_description":"Description based on the entire day's places. in a couple of words: `Historical Exploration`, `Spiritual Tour`, `Adventurous Journey`, `Dayout in a beach`,`Urban Exploration`, `Wildlife Safari`,`Relaxing Spa Day`,`Artistic Getaway`, `Romantic Getaway`, `Desert Safari`, `Island Hopping Adventure`",  "places":[{{"name":"Place Name", "description":"Place Description","time": "time to reach this place", "budget":"cost"}}, {{"name":"Place Name 2", "description":"Place Description 2","time": "time to reach this place", "budget":"cost"}}]}}, {{"day":2, "day_description": "Description based on the entire day's places. in a couple of words: `Historical Exploration`, `Spiritual Tour`, `Adventurous Journey`, `Dayout in a beach`,`Urban Exploration`, `Wildlife Safari`,`Relaxing Spa Day`,`Artistic Getaway`, `Romantic Getaway`, `Desert Safari`, `Island Hopping Adventure`", "places":[{{"name":"Place Name", "description":"Place Description","time": "time to reach this place", "budget":"cost"}}, {{"name":"Place Name 2", "description":"Place Description 2", "time": "time to reach this place", "budget":"cost"}}]}}]}}
    Note: Do not include any extra information outside this structure."""

        return query, template

    def google_place_details(self, destination, itinerary):

        # Base URLs for Google Places API
        SEARCH_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
        PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"
        
        # Load the GPlaces API key from the .env file
        load_dotenv(find_dotenv())
        api_key = os.getenv("GPLACES_API_KEY")

        json_str = re.search(r'\{.*\}', itinerary, re.DOTALL).group()
        trip_data = json.loads(json_str)
        #print(f"\nTrip data {trip_data}\n")
        # Iterate through each day and each place
        for day_data in trip_data['data']:
            for place in day_data['places']:
                # Search for the place using Google Places API
                search_payload = {
                    'input': place['name'] + ', ' + destination,
                    'inputtype': 'textquery',
                    'fields': 'place_id',
                    'key': api_key
                }
                #print(f"\nSearch payload {search_payload}\n")
                search_response = requests.get(SEARCH_URL, params=search_payload).json()
                #print(f"\nSearch response {search_response}\n")
                
                # If a place is found, get its details
                if search_response['candidates']:
                    place_id = search_response['candidates'][0]['place_id']
                    details_payload = {
                        'place_id': place_id,
                        'fields': 'name,editorial_summary,geometry,formatted_address,reviews,type,website,formatted_phone_number,price_level,rating,user_ratings_total,photo',
                        'key': api_key
                    }
                    details_response = requests.get(DETAILS_URL, params=details_payload).json()
                    place_details = details_response['result']

                    # Append the details to the original place dictionary
                    place['address'] = place_details.get('formatted_address', '')
                    place['latitude'] = place_details['geometry']['location']['lat']
                    place['longitude'] = place_details['geometry']['location']['lng']
                    place['name'] = place_details.get('name', '')
                    place['editorial_summary'] = place_details.get('editorial_summary', '')
                    place['reviews'] = place_details.get('reviews', [])
                    place['type'] = place_details.get('type', '')
                    place['website'] = place_details.get('website', '')
                    place['formatted_phone_number'] = place_details.get('formatted_phone_number', '')
                    place['price_level'] = place_details.get('price_level', '')
                    place['rating'] = place_details.get('rating', '')
                    place['user_ratings_total'] = place_details.get('user_ratings_total', '')

                    # If photos are available, get the photo URL
                    if 'photos' in place_details:
                        photo_reference = place_details['photos'][0]['photo_reference']
                        photo_payload = {
                            'maxwidth': 400,  # Can adjust the width as needed
                            'photoreference': photo_reference,
                            'key': api_key
                        }
                        place['photo_url'] = requests.get(PHOTO_URL, params=photo_payload).url

        return trip_data


    def generate_itinerary(self, llm, destination, budget, arrival_date, departure_date, start_time, end_time, additional_info):

        restaurants = main(destination)
        modified_itinerary = None  # Initialize to a default value

        if llm == "Atlas v2":
            llm_ = OpenAI(
                model_name="gpt-3.5-turbo-16k",
                temperature=0.1,
            )
            user_query, user_query_template = self.load_itinerary_template_json(
            destination, budget, arrival_date, departure_date, start_time, end_time, additional_info, restaurants
        )

            conversationchain = ConversationChain(llm=llm_, prompt=self.prompt)

            new_itinerary = conversationchain.run(user_query_template)
            
            modified_itinerary = self.google_place_details(destination=destination, itinerary=new_itinerary)
            try:
                self.log_llm_response(llm=llm, query=user_query, itinerary=modified_itinerary)
            except Exception as e:
                logging.error(f"Error: {str(e)}")

        return modified_itinerary
