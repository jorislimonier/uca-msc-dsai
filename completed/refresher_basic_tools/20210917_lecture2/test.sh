echo "entered script"
myfunc()
{
	echo "inside the function"
	x=2
}
x=1
echo $x
myfunc

echo $x
