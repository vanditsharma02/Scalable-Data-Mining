import java.text.SimpleDateFormat
import java.util.Date

//define case class for assignment_1 data
case class entry(debug_level: String, timestamp: Date, download_id: Integer, retrieval_stage: String, rest: String);

//define date and expression formats to perform mapping for assignment_1 data
val dateFormat = "yyyy-MM-dd:HH:mm:ss"
val expressionFormat = """([^\s]+), ([^\s]+)\+00:00, ghtorrent-([^\s]+) -- ([^\s]+).rb: (.*$)""".r

//load assignment_1 data into RDD
val assignment_1 = sc.textFile("ghtorrent-logs.txt").flatMap ( x => x match {
      case expressionFormat(debug_level, timestamp, download_id, retrieval_stage, rest) =>
          val df = new SimpleDateFormat(dateFormat)
          new Some(entry(debug_level, df.parse(timestamp.replace("T", ":")), download_id.toInt, retrieval_stage, rest))
      case _ => None;
})

//define case class for assignment_2 file
case class entry_2(id: Integer, url: String, owner_id: Integer, 
                   name: String, language: String, created_at: Date, forked_from: String, deleted: Integer, updated_at: Date)

//define date and expression formats to perform mapping for assignment_2 file
val dateFormat_2 = "yyyy-MM-dd HH:mm:ss"
val expressionFormat_2 = """([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)""".r

//load assignment_2 file into RDD
val assignment_2 = sc.textFile("important-repos.csv").mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter).flatMap (x => x match {
      case expressionFormat_2(id, url, owner_id, name, language, created_at, forked_from, deleted, updated_at) => {
        val df = new SimpleDateFormat(dateFormat_2)
        new Some(entry_2(id.toInt, url, owner_id.toInt, name, language, df.parse(created_at), forked_from, deleted.toInt, df.parse(updated_at)))
      }
      case _ => None;
      }).cache()

//print number of records in assignment_2 file
println("\n")
println("NUMBER OF RECORDS IN assignment_2 FILE:")
println(assignment_2.count);
println("\n")

println("NUMBER OF RECORDS IN LOG FILE REFERRING TO ENTRIES IN assignment_2 FILE:")

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

//key both RDDS
val get_assignment_1_repos = assignment_1.filter(x=>(x.rest.contains("repos"))).keyBy(_.rest).
	map(x => x.copy(_1 = x._1.split("/").slice(4,6).mkString("/").takeWhile(_ != '?').split("/", 2).last)).filter(_._1.nonEmpty); 
val get_assignment_2_repos = assignment_2.keyBy(_.name);

//join RDDs
val joined_repos = get_assignment_2_repos.join(get_assignment_1_repos); 
println(joined_repos.count);
println("\n")

println("REPOSITORIES HAVING THE MOST FAILED API CALLS [TOP 5]:")
val joined_failed = joined_repos.filter(x => x._2._2.rest.startsWith("Failed")).
           map(x => (x._1, 1)).
           reduceByKey((a,b) => a + b).
           sortBy(x => x._2, false).
           take(5)
joined_failed.foreach(tuple=>println(tuple))
println("\n")


