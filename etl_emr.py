import os
from datetime import datetime

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, year, month, dayofmonth, hour, weekofyear, dayofweek, \
    monotonically_increasing_id
from pyspark.sql.types import TimestampType, LongType, StringType


INPUT_DATA_SONG = "s3://udacity-dend/song-data"
INPUT_DATA_LOG = "s3://udacity-dend/log-data"
OUTPUT_DATA = "s3://aws-logs-265100338781-us-west-2/elasticmapreduce"
LOCAL_MODE = False
AWS_REQUIRED = False


def create_spark_session(LOCAL_MODE):
    """
    Creates the SparkSession with the necessary configurations for my local machine.
    :return: SparkSession
    """
    if LOCAL_MODE:
        spark_conf = pyspark.SparkConf() \
            .set('spark.driver.host', '127.0.0.1')
        sc = pyspark.SparkContext(master='local', appName='myAppName', conf=spark_conf)

    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-client:2.7.3,org.apache.hadoop:hadoop-aws:2.7.3,net.java.dev.jets3t:jets3t:0.9.4") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Processes the song data and creates the following tables: artists, songs.
    :param spark: SparkSession
    :param input_data: file path to the data that should be processed
    :param output_data: file path to write the result to
    """
    # get filepath to song data file
    song_data = "{}/*/*/*/*.json".format(input_data)

    # read song data file
    df = spark.read.json(song_data).dropDuplicates()

    # we have 0's in the year column that we should treat as missing values
    # print([row.year for row in df.select("year").distinct().orderBy("year").collect()])
    zero_to_null = udf(lambda x: x if x != 0 else None, LongType())
    df = df.withColumn("year", zero_to_null(df.year))
    # print([row.year for row in df.select("year").distinct().orderBy("year").collect()])

    # we have '' (empty strings) in the artist_location column that we should treat as missing values
    # print([row.artist_location for row in df.select("artist_location").distinct().orderBy("artist_location").collect()])
    empty_string_to_null = udf(lambda x: x if x else None, StringType())
    df = df.withColumn("artist_location", empty_string_to_null(df.artist_location))
    # print([row.artist_location for row in df.select("artist_location").distinct().orderBy("artist_location").collect()])

    # extract columns to create songs table
    songs_table = df.select(["song_id", "title", "artist_id", "year", "duration"]).distinct()

    print("SONGS")
    print("=====")
    print("Schema:")
    songs_table.printSchema()
    print("Row Count:")
    print(songs_table.count())
    print("Table Preview:")
    songs_table.show(n=5, truncate=False)

    # write songs table to parquet files partitioned by year and artist
    songs_table \
        .write \
        .mode("OVERWRITE") \
        .partitionBy("year", "artist_id") \
        .parquet("{}/songs/songs_table.parquet".format(output_data))

    # extract columns to create artists table
    artists_table = df \
        .select(["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]) \
        .distinct()

    print("ARTISTS")
    print("=======")
    print("Schema:")
    artists_table.printSchema()
    print("Row Count:")
    print(artists_table.count())
    print("Table Preview:")
    artists_table.show(n=5, truncate=False)

    # write artists table to parquet files
    artists_table \
        .write \
        .mode("OVERWRITE") \
        .parquet("{}/artists/artists_table.parquet".format(output_data))

    df.createOrReplaceTempView("song_data_table")


def process_log_data(spark, input_data, output_data):
    """
    Process the log data dataset and creates the user table, time table and songsplay table
    :param spark: SparkSession
    :param input_data: path/to/files to process
    :param output_data: path/to/files to write the results Datasets
    """
    # get filepath to log data file
    log_data = "{}/*/*/*events.json".format(input_data)

    # read log data file
    df = spark.read.json(log_data).dropDuplicates()

    # filter by actions for song plays
    df = df \
        .filter(df.page == "NextSong") \
        .withColumn("user_id", df["userId"].cast(LongType())) \
        .withColumnRenamed("sessionId", "session_id") \
        .withColumnRenamed("userAgent", "user_agent") \
        .withColumnRenamed("firstName", "first_name") \
        .withColumnRenamed("lastName", "last_name")

    # extract columns for users table
    user_table = df \
        .select(["first_name", "last_name", "gender", "level", "user_id"]).distinct()

    print("USERS")
    print("=====")
    print("Schema:")
    user_table.printSchema()
    print("Row Count:")
    print(user_table.count())
    print("Table Preview:")
    user_table.show(n=5, truncate=False)

    # write users table to parquet files
    user_table \
        .write \
        .mode("OVERWRITE") \
        .parquet("{}/users/users_table.parquet".format(output_data))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(col("ts")))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S'), StringType())
    df = df.withColumn("start_time", get_datetime(col("ts")))

    # extract columns to create time table
    df = df.withColumn("hour", hour("timestamp"))
    df = df.withColumn("day", dayofmonth("timestamp"))
    df = df.withColumn("month", month("timestamp"))
    df = df.withColumn("year", year("timestamp"))
    df = df.withColumn("week", weekofyear("timestamp"))
    df = df.withColumn("weekday", dayofweek("timestamp"))

    time_table = df \
        .select(["start_time", "hour", "day", "week", "month", "year", "weekday"]) \
        .distinct()

    print("TIME")
    print("====")
    print("Schema:")
    time_table.printSchema()
    print("Row Count:")
    print(time_table.count())
    print("Table Preview:")
    time_table.show(n=5, truncate=False)

    # write time table to parquet files partitioned by year and month
    time_table \
        .write \
        .mode("OVERWRITE") \
        .partitionBy("year", "month") \
        .parquet("{}/time/time_table.parquet".format(output_data))

    # read in song data to use for songplays table
    song_df = spark.sql("SELECT DISTINCT song_id, artist_id, artist_name FROM song_data_table")

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(song_df, song_df.artist_name == df.artist, "inner") \
        .distinct() \
        .withColumn("songplay_id", monotonically_increasing_id()) \
        .select(["songplay_id", "start_time", "user_id", "level", "session_id", "location", "user_agent", "song_id",
                 "artist_id"])

    print("SONGPLAYS")
    print("=========")
    print("Schema:")
    songplays_table.printSchema()
    print("Row Count:")
    print(songplays_table.count())
    print("Table Preview:")
    songplays_table.show(n=5, truncate=False)

    # write songplays table to parquet files partitioned by year and month
    songplays_table \
        .write \
        .mode("OVERWRITE") \
        .partitionBy("year", "month") \
        .parquet("{}/songplays/songplays_table.parquet".format(output_data))


def main():
    """
    Main Function
    """

    start = datetime.now()
    print("{}: (1/5) Starting ETL job...".format(start))
    print("\tLOCAL_MODE is set to {}.".format(LOCAL_MODE))
    print("\tLoading song data from {}.".format(INPUT_DATA_SONG))
    print("\tLoading log data from {}.".format(INPUT_DATA_LOG))
    print("\tSaving data to {}.".format(OUTPUT_DATA))

    if AWS_REQUIRED:
        print("\tSetting AWS credentials from dl.cfg...")
        os.environ['AWS_ACCESS_KEY_ID'] = conf.get('AWS', 'AWS_ACCESS_KEY_ID')
        os.environ['AWS_SECRET_ACCESS_KEY'] = conf.get('AWS', 'AWS_SECRET_ACCESS_KEY')
        print("\tDone.")
    else:
        print("\tYou specified that AWS credentials are unnecessary. Proceeding...")

    print("{}: (2/5) Creating Spark session...".format(datetime.now()))
    spark = create_spark_session(LOCAL_MODE)

    print("{}: (3/5) Processing song data...".format(datetime.now()))
    process_song_data(spark, INPUT_DATA_SONG, OUTPUT_DATA)
    print("{}: (4/5) Processing log data...".format(datetime.now()))
    process_log_data(spark, INPUT_DATA_LOG, OUTPUT_DATA)

    end = datetime.now()
    print("{}: (5/5) ETL job finished. Elapsed time: {}.".format(end, end-start))


if __name__ == "__main__":
    main()
