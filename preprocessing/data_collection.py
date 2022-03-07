import netCDF4
import json
import sys
import numpy as np
import os
import csv
import random
import datetime
np.set_printoptions(threshold=sys.maxsize)


# CONST
BASE_PATH = "F:\\Tornado\\"
FOLDER_OUTPUTS = "F:\\Tornado\\output\\"
FOLDER_WEATHER = "weather\\"
FOLDER_EVENTS = "events\\"

MIN_YEAR = 2015
MAX_YEAR = 2018


LON_RANGE = 2.5
LAT_RANGE = 2.5
DAY_RANGE = 6

DAMAGE_LIMIT = 240000
ALLOWED_EVENTS = ["Tornado"]

RANDOM_LATLON_INDEXES = 15

def readOtherWeather(eventsDates, eventsLatitudes, eventsLongitudes):
    otherWeatherDataset = {}
    otherWeatherDatesRanges = {}
    
    for y in range(MIN_YEAR,MAX_YEAR+1):
        for date in eventsDates[y]:
            month = int(date[4:6])
            day = int(date[6:])
            if day - ((DAY_RANGE-1)*2) < 1: # Check 10 days before
                continue
            dateConflict = False
            for i in range(day - ((DAY_RANGE-1)*2), day):
                for tmpDate in eventsDates[y]:
                    tmpMonth = int(tmpDate[4:6])
                    if(tmpMonth != month):
                       continue
                    tmpDay = int(tmpDate[6:]) 
                    if(tmpDay == i): # There is another event near the considered one                 
                        dateConflict = True
                        break
                if dateConflict: # Stop to search, the date is not available
                    break
            if dateConflict:
                dateConflict = False
                continue
            
            if not y in otherWeatherDatesRanges:
                otherWeatherDatesRanges[y] = []
            
            dateRange = [day-DAY_RANGE,month]
            if(not dateRange in otherWeatherDatesRanges[y]):
                otherWeatherDatesRanges[y].append(dateRange)
            

    


    goodEventId = 0
    for y in range(MIN_YEAR, MAX_YEAR + 1):
        print("Reading weather for year "+str(y))
        '''
        weatherFile = BASE_PATH + FOLDER_WEATHER + str(y) + ".nc"
        nc = netCDF4.Dataset(weatherFile, mode='r')
        time = nc.variables['time']  
        latitudes = nc.variables['latitude']
        longitudes = nc.variables['longitude']
        
        dates = netCDF4.num2date(time[:], time.units, time.calendar)
        print(len(otherWeatherDatesRanges[y]))
        '''
        counter = 0
        for dateRange in otherWeatherDatesRanges[y]:
            
            day = dateRange[0]
            month = dateRange[1]
            randomIndexes = RANDOM_LATLON_INDEXES
            if RANDOM_LATLON_INDEXES > len(eventsLatitudes[y]):
                randomIndexes = len(eventsLatitudes[y])
            randomIndexes = random.sample(range(0,len(eventsLatitudes[y])), randomIndexes)
            if day <= DAY_RANGE:
                continue
            counter += 1
            continue
            for k in randomIndexes: # Take 4 examples of latitudes and longitudes situation for each range of dates without tornados in the world
                eventLat = float(eventsLatitudes[y][k])
                eventLon = float(eventsLongitudes[y][k]) + 180

                print("Reading good weather in area ("+ str(eventLat) +","+ str(eventLon)+ ")  in date: "+ str(day) + "/" + str(month) + "/" + str(y))

                latIndexes = [i for i, latitude in enumerate(latitudes) if checkLatitude(eventLat,latitude)]
                lonIndexes = [i for i, longitude in enumerate(longitudes) if checkLongitude(eventLon,longitude)]
                dateIndexes = [i for i, date in enumerate(dates) if date.year == y and date.month == month and date.day in range(day-DAY_RANGE,day)]
                
                minDateIndex = dateIndexes[0]
                maxDateIndex = dateIndexes[-1]
                minLatIndex = latIndexes[0]
                maxLatIndex = latIndexes[-1]
                

                minLonIndex = lonIndexes[0]
                maxLonIndex = lonIndexes[-1] 
                indexes = {
                    "latitude": np.asarray(latitudes[minLatIndex:maxLatIndex]).tolist(),
                    "longitude": np.asarray(longitudes[minLonIndex:maxLonIndex]).tolist()
                }
                weatherData = {   
                            "wind100m_u": np.asarray(nc.variables['u100'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,       
                            "wind100m_v": np.asarray(nc.variables['v100'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,               
                            "wind10m_u": np.asarray(nc.variables['u10'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,
                            "wind10m_v": np.asarray(nc.variables['v10'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,
                            "temperature": np.asarray(nc.variables['t2m'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,              
                            "precipitation_rate": np.asarray(nc.variables['mtpr'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,              
                            "precipitation": np.asarray(nc.variables['tp'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,              
                            "column_rain": np.asarray(nc.variables['tcrw'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,
                            "cloud_cover": np.asarray(nc.variables['tcc'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,                               
                            "large_scale_rain_cover": np.asarray(nc.variables['lsrr'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() 
                }
                eventInfo = {
                    "name": goodEventId,
                    "date": str(day)+"/"+str(month)+"/"+str(y),
                    "latitude": eventLat,
                    "longitude": eventLon
                }
                eventDict = {
                    "info": eventInfo,
                    "coordinates": indexes,
                    "data": weatherData
                }
                otherWeatherDataset[goodEventId] = eventDict
                goodEventId+=1
                
    print(counter)
    return otherWeatherDataset

            
                
                


# Read all the weather data
def readEventsWeather(events):
    tornadoDataset = {}

    eventsNumber = sum(len(lst) for lst in events.values())
    for y in range(MIN_YEAR,MAX_YEAR+1):
        
        print("Searching for weather in " + str(y) + "...")
   
        weatherFile = BASE_PATH + FOLDER_WEATHER + str(y) + ".nc"
        nc = netCDF4.Dataset(weatherFile, mode='r')
        time = nc.variables['time']  
        latitudes = nc.variables['latitude']
        longitudes = nc.variables['longitude']

        dates = netCDF4.num2date(time[:], time.units, time.calendar)
        readedEvents = 0

        for eventType in ALLOWED_EVENTS:
            if eventType not in events:
                continue
            currentEvents = events[eventType]
            print("Searching data for event type: "+eventType)
        
            for event in currentEvents:
                readedEvents += 1
                year = int(event["date"][:4])
                month = int(event["date"][4:6])
                day = int(event["date"][6:])

                if day - DAY_RANGE <= 0:
                    continue
                if not year == y:
                    continue

                print("Searching data for event id: "+ event["id"] + " ["+str(readedEvents)+"/"+str(eventsNumber)+"]")
                
                dateIndexes = [i for i, date in enumerate(dates) if date.year == year and date.month == month and date.day in range(day-DAY_RANGE,day)]

                eventLat = float(event["latitude"])
                eventLon = float(event["longitude"])+180
            
       
                latIndexes = [i for i, latitude in enumerate(latitudes) if checkLatitude(eventLat,latitude)]
                lonIndexes = [i for i, longitude in enumerate(longitudes) if checkLongitude(eventLon,longitude)]
                minDateIndex = dateIndexes[0]
                maxDateIndex = dateIndexes[-1]

                minLatIndex = latIndexes[0]
                maxLatIndex = latIndexes[-1]

                minLonIndex = lonIndexes[0]
                maxLonIndex = lonIndexes[-1] 

                indexes = {
                    "latitude": np.asarray(latitudes[minLatIndex:maxLatIndex]).tolist(),
                    "longitude": np.asarray(longitudes[minLonIndex:maxLonIndex]).tolist()
                }
                currentDataset = None 
  
                if eventType == "Tornado":        
                    weatherData = {   
                        "wind100m_u": np.asarray(nc.variables['u100'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,       
                        "wind100m_v": np.asarray(nc.variables['v100'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,               
                        "wind10m_u": np.asarray(nc.variables['u10'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,
                        "wind10m_v": np.asarray(nc.variables['v10'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,
                        "temperature": np.asarray(nc.variables['t2m'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,              
                        "precipitation_rate": np.asarray(nc.variables['mtpr'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,              
                        "precipitation": np.asarray(nc.variables['tp'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,              
                        "column_rain": np.asarray(nc.variables['tcrw'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,
                        "cloud_cover": np.asarray(nc.variables['tcc'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() ,                               
                        "large_scale_rain_cover": np.asarray(nc.variables['lsrr'][minDateIndex:maxDateIndex,minLatIndex:maxLatIndex,minLonIndex:maxLonIndex]).tolist() 
                    }
                    currentDataset = tornadoDataset
               
                 
                eventInfo = {
                    "name": event["id"],
                    "date": str(day)+"/"+str(month)+"/"+str(year),
                    "latitude": eventLat,
                    "longitude": eventLon
                }
                eventDict = {
                    "info": eventInfo,
                    "coordinates": indexes,
                    "data": weatherData
                }
                currentDataset[event["id"]] = eventDict
      
    
    return tornadoDataset

# Check if weather data are in range zone
def checkLatitude(hurricaneLat, lat):
    return (lat <= hurricaneLat + LAT_RANGE and lat >= hurricaneLat - LAT_RANGE)

def checkLongitude(hurricaneLon, lon):
    return (lon <= hurricaneLon + LON_RANGE and lon >= hurricaneLon - LON_RANGE)

def sanitizeDamage(damage):

    damage = damage.replace("K","")
    try:
        damage = float(damage)*1000
    except:
        damage = 0
    return damage

# Used to remove useless months from years when download the weather data
def printCountEvents():
    for year in range(MIN_YEAR,MAX_YEAR+1):
        with open(BASE_PATH + FOLDER_EVENTS + str(year) + ".csv", newline='') as csvfile:
            data = list(csv.reader(csvfile))
            counters = {}
            i = 0
            for row in data:
                if((not i==0) and sanitizeDamage(row[24])+sanitizeDamage(row[25]) > DAMAGE_LIMIT):
                    if(row[12] in counters):
                        counters[row[12]] += 1
                    else:
                        counters[row[12]] = 1
                i += 1
                
def readEvents():
    events = {}
    dates = {}
    latitudes = {}
    longitudes = {}
    for year in range(MIN_YEAR,MAX_YEAR+1):
        print("Reading events in year: "+str(year))
        with open(BASE_PATH + FOLDER_EVENTS + str(year) + ".csv", newline='') as csvfile:
            data = list(csv.reader(csvfile))
   
            i = 0
            discardedEvents = 0
            readedEvents = 0
            for row in data:
                if((not i==0) and row[12] in ALLOWED_EVENTS and sanitizeDamage(row[24])+sanitizeDamage(row[25]) > DAMAGE_LIMIT):
                
                    date = row[1]
                    if(int(date)<=9):
                        date = row[0] + "0" + date
                    else:
                        date = row[0] + date
                    
                   
                    if(row[44] == "" or row[45] == ""):
                        discardedEvents += 1
                        continue
                    if year not in dates: 
                        dates[year] = []
                        longitudes[year] = []
                        latitudes[year] = []
                    
                    dates[year].append(date)
                    latitudes[year].append(row[44])
                    longitudes[year].append(row[45])

                    eventRow = {
                        "id": row[7],
                        "date": date,
                        "type": row[12],
                        "latitude": row[44],
                        "longitude": row[45]
                    }
                    
                    if row[12] not in events: 
                        events[row[12]] = []
                    
                    events[row[12]].append(eventRow)
                    readedEvents += 1
                i += 1
            print("Events readed correctly: "+ str(readedEvents))
            print("Events discarded: "+ str(discardedEvents))
    return dates,latitudes, longitudes, events

    
def main():
    start_time = datetime.datetime.now()

    print("Reading generic events ...")
    dates,latitudes,longitudes,events = readEvents()
    print("Reading good weather data ...")
    goodEventsDataset = readOtherWeather(dates, latitudes, longitudes)
    print(len(goodEventsDataset))
    print("Reading weather event data ...")
    tornadoDataset = readEventsWeather(events)
    
    print("Generating tornado output ...")
    mongoDataset = formatMongoString(tornadoDataset)
    f = open(FOLDER_OUTPUTS + "tornado-mongo.json", "w+")
    f.write(mongoDataset)
    f.close()

    print("Generating good events output ...")
    writeMongoString(goodEventsDataset)

    print("Operation completed. Execution time: "+str(datetime.datetime.now()-start_time))
   
main()
