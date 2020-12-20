function watchform() {
    let posttodo = document.getElementById("posttodo");
    let newtodo = document.getElementById("newtodo");
    let listdetails = document.getElementById("listdetails")


    posttodo.addEventListener("click", (event) => {
        event.preventDefault();
        var div = document.createElement("div");
        div.className = "divv";
        var todo = document.createTextNode(newtodo.value);
        $.ajax({
            type: "GET",
            url: "./predict.py",
            data: { param: todo}
          }).done(function( o ) {
             // do something
          });
        div.appendChild(todo);


        if (newtodo.value === "") {
            listdetails.textContent = listdetails.textContent
        }
        else {
            listdetails.appendChild(div);
            newtodo.value = ""
        }
    })
}

watchform()