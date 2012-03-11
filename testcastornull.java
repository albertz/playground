public class testcastornull {

    public static void main(String[] args) {
        Integer o = castOrNull("hello", Integer.class);
		System.out.println(o);
    }

	public static <T> T castOrNull(Object obj, Class<T> clazz) {
    	if(clazz.isInstance(obj))
        	return clazz.cast(obj);
		return null;
	}
}

