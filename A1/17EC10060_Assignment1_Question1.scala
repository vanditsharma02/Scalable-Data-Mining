import java.text.SimpleDateFormat
import java.util.Date

//define case class
case class entry(debug_level: String, timestamp: Date, download_id: Integer, retrieval_stage: String, rest: String);

//define date and expression formats to perform mapping
val dateFormat = "yyyy-MM-dd:HH:mm:ss"
val expressionFormat = """([^\s]+), ([^\s]+)\+00:00, ghtorrent-([^\s]+) -- ([^\s]+).rb: (.*$)""".r

//load data into RDD
val assignment_1 = sc.textFile("ghtorrent-logs.txt").flatMap ( x => x match {
      case expressionFormat(debug_level, timestamp, download_id, retrieval_stage, rest) =>
          val df = new SimpleDateFormat(dateFormat)
          new Some(entry(debug_level, df.parse(timestamp.replace("T", ":")), download_id.toInt, retrieval_stage, rest))
      case _ => None;
})

//print number of lines in RDD
println("\n")
println("NUMBER OF LINES IN RDD:");
println(assignment_1.count);
println("\n")

//print number of warn messages
println("NUMBER OF WARN MESSAGES:")
println(assignment_1.filter(x => x.debug_level == "WARN").count);
println("\n")

//print number of repositories processed in total when the retrieval_stage is “api_client”
println("NUMBER OF REPOSITORIES PROCESSED WHEN retrieval_stage IS api_client:")

//define a custom mapping function to obtain name of two different kinds of repositories (containing keywords "Repo" and "repos" respectively)
def get_repo_name(x:entry):String={
	val containsRepo = x.rest.contains("Repo")
	try{
		if(containsRepo){
			val temp = x.rest.split(" ")
			val repo = temp(temp.indexOf("Repo")+1)
			if(repo.contains("/")){
				return repo
			}
			else{
				return null
			}
		}
		else{
			var temp = x.rest.split("github.com/repos/")(1)
			temp = temp.split("\\?")(0)
			var temp1 = temp.split("/")
			if(temp1.size>1){
				val repo=temp1(0)+"/"+temp1(1)
				return repo
			}
			else{	
				return null
			}
		}
		return null
	}
	catch{
		case error:Exception=>return null
	}
}

//obtain a list of repositories with api_client retrieval stage (after cleaning) along with their activity count
var api_client_repos = assignment_1.filter(_.retrieval_stage == "api_client").filter(x=>(x.rest.contains("Repo") || x.rest.contains("repos"))).
	map(x=>get_repo_name(x)).filter(x=>x!=null).map(x=>(x,1)).reduceByKey(_+_)
println(api_client_repos.distinct.count)
println("\n")

//print top 5 clients with most number of http requests
println("CLIENTS WITH THE MOST HTTP REQUESTS [TOP 5]:")
var clients = assignment_1.filter(_.retrieval_stage == "api_client").
    keyBy(_.download_id).
    mapValues(x => 1).
    reduceByKey((a,b) => a + b).
    sortBy(x => x._2, false).
    take(5)
clients.foreach(tuple=>println(tuple))
println("\n")

//print top 5 clients with most number of failed http requests
println("CLIENTS WITH THE MOST FAILED HTTP REQUESTS [TOP 5]:")
clients = assignment_1.filter(_.retrieval_stage == "api_client").
    filter(_.rest.startsWith("Failed")).
    keyBy(_.download_id).
    mapValues(x => 1).
    reduceByKey((a,b) => a + b).
    sortBy(x => x._2, false).
    take(5)
clients.foreach(tuple=>println(tuple))
println("\n")

//print most active hour of the day
println("MOST ACTIVE HOUR OF THE DAY:")
val most_active_hour = assignment_1.keyBy(_.timestamp.getHours).
    mapValues(x => 1).
    reduceByKey((a,b) => a + b).
    sortBy(x => x._2, false).
    take(1)
most_active_hour.foreach(tuple=>println(tuple))
println("\n")

println("MOST ACTIVE REPOSITORY:") 
//obtain a list of all repositories (after cleaning) along with their activity count
var all_repos = assignment_1.filter(x=>(x.rest.contains("Repo") || x.rest.contains("repos"))).
	map(x=>get_repo_name(x)).filter(x=>x!=null).map(x=>(x,1)).reduceByKey(_+_)

//sort and print the repository with most activity count
val most_active_repo1 = all_repos.
    sortBy(x => x._2, false).
    take(1)
most_active_repo1.foreach(tuple=>println(tuple))
println("\n")

//print top 5 access keys failing most often
println("ACCESS KEYS FAILING MOST OFTEN [TOP 5]:")
val failing_keys = assignment_1.filter(_.rest.startsWith("Failed")).
    filter(_.rest.contains("Access: ")).
    map(_.rest.split("Access: ", 2)(1).split(",", 2)(0)).
    map(x => (x, 1)).
    reduceByKey((a,b) => a + b).
    sortBy(x => x._2, false).
    take(5)
failing_keys.foreach(tuple=>println(tuple))
println("\n")




