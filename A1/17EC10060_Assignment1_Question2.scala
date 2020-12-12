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

//define a function to compute inverted index given an RDD and a field
def inverted_index(rdd : org.apache.spark.rdd.RDD[entry], x : String): 
	org.apache.spark.rdd.RDD[(Any, Iterable[entry])] = {
	return rdd.groupBy(r=>x match {
		case "debug_level" => r.debug_level
		case "timestamp" => r.timestamp
		case "download_id" => r.download_id
		case "retrieval_stage" => r.retrieval_stage
	})
}

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

//obtain a list of all repositories (after cleaning) accessed by the client 'ghtorrent-22' (client id = 22)
println("\n")
println("NUMBER OF DIFFERENT REPOSITORIES ACCESSED BY CLIENT 'ghtorrent-22' (WITHOUT INVERTED INDEX):")
var repos_withoutInvertedIndex = assignment_1.filter(_.download_id == 22).filter(x=>(x.rest.contains("Repo") || x.rest.contains("repos"))).
	map(x=>get_repo_name(x)).filter(x=>x!=null)
println(repos_withoutInvertedIndex.distinct.count)
println("\n")

println("NUMBER OF DIFFERENT REPOSITORIES ACCESSED BY CLIENT 'ghtorrent-22' (WITH INVERTED INDEX):")
val created_inverted_index = inverted_index(assignment_1, "download_id");
val postings = created_inverted_index.lookup(22)
val iter = Iterator(postings).next();
var repos_withInvertedIndex = List[String]();

for (x <- iter){
    	for (y <- x) {
		if(y.rest.contains("Repo") || y.rest.contains("repos")){
			if (!repos_withInvertedIndex.contains(get_repo_name(y))) {
    		      		repos_withInvertedIndex = repos_withInvertedIndex :+ get_repo_name(y);
			}
        	}	
    	}
}
println(repos_withInvertedIndex.size);
println("\n")


